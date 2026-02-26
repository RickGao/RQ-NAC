import os
import time
import logging
import statistics
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from torch.utils.data import DataLoader

from rqvae.img_datasets import create_dataset
from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.models import create_model


def load_model(path, ema=False):
    model_config = os.path.join(os.path.dirname(path), 'config.yaml')
    config = load_config(model_config)
    config.arch = augment_arch_defaults(config.arch)

    model, _ = create_model(config.arch, ema=False)
    ckpt = torch.load(path, weights_only=False)
    state_key = 'state_dict_ema' if ema else 'state_dict'
    model.load_state_dict(ckpt[state_key])

    return model, config



def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_fname = os.path.join(log_dir, 'profile.log')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_fname),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


def _run_compress_decode(mod, xs, device, start_ev, end_ev):
    """Run encode+quantize (compress) and decode (recover), return (t_compress_ms, t_decode_ms)."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_ev.record()
        z_e = mod.encode(xs)
        z_q, _, _ = mod.quantizer(z_e)
        end_ev.record()
        torch.cuda.synchronize()
        t_compress_ms = start_ev.elapsed_time(end_ev)

        start_ev.record()
        _ = mod.decode(z_q)
        end_ev.record()
        torch.cuda.synchronize()
        t_decode_ms = start_ev.elapsed_time(end_ev)
    else:
        t0 = time.perf_counter()
        z_e = mod.encode(xs)
        z_q, _, _ = mod.quantizer(z_e)
        t1 = time.perf_counter()
        t_compress_ms = (t1 - t0) * 1000.0

        t0 = time.perf_counter()
        _ = mod.decode(z_q)
        t1 = time.perf_counter()
        t_decode_ms = (t1 - t0) * 1000.0
    return t_compress_ms, t_decode_ms


@torch.no_grad()
def profile_reconstruction(
    dataset,
    stage1_model,
    device,
    batch_size: int = 1,
    num_workers: int = 4,
    n_warmup: int = 10,
    max_iters: int | None = None,
    per_sample: bool = False,
    logger: logging.Logger | None = None,
):

    mod = stage1_model.module if hasattr(stage1_model, 'module') else stage1_model

    loader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    stage1_model.eval()

    # 预热，不记时间
    it_warmup = 0
    for xs, _ in loader:
        xs = xs.to(device, non_blocking=True)
        if per_sample:
            for i in range(xs.size(0)):
                _ = stage1_model(xs[i:i+1])[0]
        else:
            _ = stage1_model(xs)[0]
        it_warmup += 1
        if it_warmup >= n_warmup:
            break

    if device.type == 'cuda':
        torch.cuda.synchronize()

    total_imgs = 0
    total_time_compress_ms = 0.0
    total_time_decode_ms = 0.0

    if device.type == 'cuda':
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
    else:
        ev_start = ev_end = None

    it_count = 0
    iter_times_ms = []
    for xs, _ in loader:
        xs = xs.to(device, non_blocking=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        bsz = xs.size(0)

        if per_sample:
            t_compress_ms = 0.0
            t_decode_ms = 0.0
            for i in range(bsz):
                tc, td = _run_compress_decode(mod, xs[i:i+1], device, ev_start, ev_end)
                t_compress_ms += tc
                t_decode_ms += td
        else:
            t_compress_ms, t_decode_ms = _run_compress_decode(mod, xs, device, ev_start, ev_end)

        total_time_compress_ms += t_compress_ms
        total_time_decode_ms += t_decode_ms
        total_imgs += bsz
        it_count += 1

        t_total_ms = t_compress_ms + t_decode_ms
        iter_times_ms.append(t_total_ms)
        if logger is not None and it_count % 10 == 0:
            logger.info(
                f"iter {it_count}, batch_size={bsz}, "
                f"compress={t_compress_ms:.4f} ms, decode={t_decode_ms:.4f} ms, total={t_total_ms:.4f} ms"
            )

        if max_iters is not None and it_count >= max_iters:
            break

    n = max(total_imgs, 1)
    total_time_s = (total_time_compress_ms + total_time_decode_ms) / 1000.0
    avg_compress_ms = total_time_compress_ms / n
    avg_decode_ms = total_time_decode_ms / n
    avg_total_ms = avg_compress_ms + avg_decode_ms
    imgs_per_sec = total_imgs / max(total_time_s, 1e-8)

    iter_min_ms = min(iter_times_ms) if iter_times_ms else 0.0
    iter_max_ms = max(iter_times_ms) if iter_times_ms else 0.0
    iter_std_ms = statistics.stdev(iter_times_ms) if len(iter_times_ms) > 1 else 0.0

    result = {
        "images": total_imgs,
        "seconds": total_time_s,
        "imgs_per_sec": imgs_per_sec,
        "avg_ms_per_image": avg_total_ms,
        "avg_compress_ms": avg_compress_ms,
        "avg_decode_ms": avg_decode_ms,
        "total_compress_ms": total_time_compress_ms,
        "total_decode_ms": total_time_decode_ms,
        "iter_min_ms": iter_min_ms,
        "iter_max_ms": iter_max_ms,
        "iter_std_ms": iter_std_ms,
    }

    return result


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vqvae', type=str, required=True,
                        help='path to vqvae/rqvae checkpoint (epochXXX_model.pt)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size for profiling (default: 1)')
    parser.add_argument('--split', type=str, default='val',
                        help='dataset split: val or train')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of DataLoader workers')
    parser.add_argument('--warmup-iters', type=int, default=10,
                        help='warmup iterations not counted in timing')
    parser.add_argument('--max-iters', type=int, default=None,
                        help='max iterations to profile (None = all)')
    parser.add_argument('--per-sample', action='store_true',
                        help='split each batch into single samples (imgs[i:i+1])')

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    vqvae_model, config = load_model(args.vqvae)
    vqvae_model = vqvae_model.to(device)
    vqvae_model = torch.nn.DataParallel(vqvae_model).eval()

    ckpt_dir = os.path.dirname(args.vqvae)
    profile_dir = os.path.join(ckpt_dir, 'profile')
    logger = setup_logger(profile_dir)

    logger.info(f"device: {device}")
    logger.info(f"ckpt: {args.vqvae}")
    logger.info(f"batch_size: {args.batch_size}, per_sample={args.per_sample}")

    dataset_trn, dataset_val = create_dataset(config, is_eval=True, logger=logger)
    dataset = dataset_val if args.split in ['val', 'valid'] else dataset_trn
    logger.info(f"profiling on {config.dataset.type}/{args.split}, size={len(dataset)}")

    # Profiling
    result = profile_reconstruction(
        dataset=dataset,
        stage1_model=vqvae_model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_warmup=args.warmup_iters,
        max_iters=args.max_iters,
        per_sample=args.per_sample,
        logger=logger,
    )

    logger.info("=" * 60)
    logger.info("RECONSTRUCTION INFERENCE PROFILING")
    logger.info("=" * 60)
    logger.info(f"images processed : {result['images']}")
    logger.info(f"total time (s)   : {result['seconds']:.3f}")
    logger.info(f"images / second  : {result['imgs_per_sec']:.2f}")
    logger.info("")
    logger.info("Per-image averages (ms):")
    logger.info(f"  compress (encode+quantize) : {result['avg_compress_ms']:.4f} ms")
    logger.info(f"  decode (recover)          : {result['avg_decode_ms']:.4f} ms")
    logger.info(f"  total                     : {result['avg_ms_per_image']:.4f} ms")
    logger.info("")
    logger.info("Per-iter variance (diagnostic):")
    logger.info(f"  min={result['iter_min_ms']:.4f} ms, max={result['iter_max_ms']:.4f} ms, std={result['iter_std_ms']:.4f} ms")
    logger.info("  (if min==max and std≈0, timing may be quantized/blocked)")
    logger.info("=" * 60)
    logger.info(f"log saved to: {os.path.join(profile_dir, 'profile.log')}")


if __name__ == '__main__':
    main()
