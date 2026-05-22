from ngram import NGramModel
from arithmetic_coding import (
    ArithmeticEncoder,
    profile_encode_decode,
    profile_stats_lines,
    _format_bytes,
)
import sys, logging, os

import psutil


def readcode(filename, n=None):
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            line = line.strip()
            if line:
                numbers = [int(x) for x in line.split()]
                result.append(numbers)
    return result

# N-GRAM
N = 2
# K smoothing
K = 0.1
# Depth of RQVAE code
D = 8



logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.makedirs("logs", exist_ok=True)

logfile = "logs/"+str(N)+"gram_x"+str(D)+"_log.txt"
# logfile = "logs/"+str(N)+"gram_x"+str(D)+"_20k_log.txt"
print("Log:", logfile)
file_handler = logging.FileHandler(logfile, mode='w', encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


filename = "data/codes23x40x"+ str(D) +".txt"
# filename = "data/codes23x40x"+ str(D) +"-20k.txt"

logger.info(f"N={N}, K={K}")
logger.info(f"Data: {filename}")

training_sequences = readcode(filename, 900)


model = NGramModel(n=N, k=K, start_token=-1, end_token=-2)

model.fit(training_sequences)

model.save("models/"+str(N)+"gram_x"+str(D)+".pkl")
# model.save("models/"+str(N)+"gram_x"+str(D)+"_20k.pkl")


# model = NGramModel.load("models/"+str(N)+"gram_x"+str(D)+".pkl")


encoder = ArithmeticEncoder(ngram_model=model, bits=32)


print("Encoder Created")

codes = readcode(filename, 1000)[900:1000]

avgrate = []
encode_times_ms = []
decode_times_ms = []
encode_cpu_times_ms = []
decode_cpu_times_ms = []
mem_after_decode_bytes = []
mem_delta_bytes = []

for i in range(len(codes)):
    logger.info(f"Code: {i}")
    test_sequence = codes[i]

    stats = profile_encode_decode(encoder, test_sequence)
    encoded_bits = stats["encoded"]
    decoded_sequence = stats["decoded"]

    encode_ms = stats["encode_wall_s"] * 1000.0
    decode_ms = stats["decode_wall_s"] * 1000.0
    encode_times_ms.append(encode_ms)
    decode_times_ms.append(decode_ms)
    encode_cpu_times_ms.append(stats["encode_cpu_total_s"] * 1000.0)
    decode_cpu_times_ms.append(stats["decode_cpu_total_s"] * 1000.0)
    mem_after_decode_bytes.append(stats["mem_after_decode_bytes"])
    mem_delta_bytes.append(stats["mem_delta_bytes"])

    logger.info(f"Encoded: {len(encoded_bits)} bits")
    rate = len(encoded_bits) / (len(test_sequence) * 11)
    avgrate.append(rate)
    logger.info(f"Compression Rate: {rate:.2%}")
    logger.info(f"Verification: {'Correct' if decoded_sequence == test_sequence else 'Wrong'}")
    for line in profile_stats_lines(stats):
        logger.info(line)

logger.info(f"Average Compression Rate: {sum(avgrate)/len(avgrate):.2%}")
logger.info("")
logger.info("Timing summary (ms):")
avg_encode = sum(encode_times_ms) / len(encode_times_ms)
avg_decode = sum(decode_times_ms) / len(decode_times_ms)
logger.info(f"  Average encode time: {avg_encode:.4f} ms")
logger.info(f"  Average decode time: {avg_decode:.4f} ms")
logger.info(f"  encode: min={min(encode_times_ms):.4f}, max={max(encode_times_ms):.4f}")
logger.info(f"  decode: min={min(decode_times_ms):.4f}, max={max(decode_times_ms):.4f}")
avg_encode_cpu = sum(encode_cpu_times_ms) / len(encode_cpu_times_ms)
avg_decode_cpu = sum(decode_cpu_times_ms) / len(decode_cpu_times_ms)
logger.info("")
logger.info("CPU time summary (ms):")
logger.info(f"  Average encode CPU: {avg_encode_cpu:.4f} ms")
logger.info(f"  Average decode CPU: {avg_decode_cpu:.4f} ms")
logger.info(f"  encode CPU: min={min(encode_cpu_times_ms):.4f}, max={max(encode_cpu_times_ms):.4f}")
logger.info(f"  decode CPU: min={min(decode_cpu_times_ms):.4f}, max={max(decode_cpu_times_ms):.4f}")
logger.info("")
logger.info("Memory RSS summary:")
logger.info(f"  Process RSS now: {_format_bytes(psutil.Process().memory_info().rss)}")
logger.info(f"  Peak RSS after decode (per sample): {_format_bytes(max(mem_after_decode_bytes))}")
logger.info(f"  Max RSS delta (per sample): {_format_bytes(max(mem_delta_bytes))}")




