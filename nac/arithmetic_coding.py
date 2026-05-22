import time
from typing import Dict, Tuple, List, Set

import psutil


class ArithmeticEncoder:
    """
    Arithmetic Encoder based on a standard implementation.
    Supports static or N-gram conditional probability distributions.
    """

    __slots__ = [
        'bits', 'start_token', 'end_token', 'EOM', 'prob_distribution',
        'n', '_vocab', 'TOP_VALUE', 'FIRST_QUARTER', 'HALF',
        'THIRD_QUARTER', '_ranges_cache', '_sorted_symbols_cache'
    ]

    def __init__(self, ngram_model=None, prob_distribution=None, bits=32,
                 start_token=-1, end_token=-2):
        """
        Initialize the encoder with either an N-gram model or a static probability table.
        """
        self.bits = bits
        self.start_token = start_token
        self.end_token = end_token
        self.EOM = end_token

        # Load probability distribution from N-gram model or direct input
        if ngram_model is not None:
            self.prob_distribution = ngram_model.get_probability_distribution()
            self.n = ngram_model.n
            self._vocab = ngram_model.vocab
        elif prob_distribution is not None:
            self.prob_distribution = prob_distribution
            if self.prob_distribution:
                sample_context = next(iter(self.prob_distribution.keys()))
                self.n = len(sample_context) + 1
            else:
                self.n = 2
            self._vocab = None
        else:
            raise ValueError("Must provide either ngram_model or prob_distribution")

        # Bit precision constants
        self.TOP_VALUE = (1 << self.bits) - 1
        self.FIRST_QUARTER = (self.TOP_VALUE >> 2) + 1
        self.HALF = self.FIRST_QUARTER * 2
        self.THIRD_QUARTER = self.FIRST_QUARTER * 3

        # Cache for efficiency
        self._ranges_cache = {}
        self._sorted_symbols_cache = {}

        # Precision check
        max_vocab_size = len(self.vocab)
        max_safe_count = int((self.TOP_VALUE + 1) / 4) + 1
        if max_vocab_size > max_safe_count:
            raise ValueError(
                f"Insufficient precision: vocabulary size {max_vocab_size} "
                f"exceeds safe limit {max_safe_count}. "
                f"Increase bits (current: {bits})"
            )

    @property
    def vocab(self) -> Set[int]:
        """
        Get vocabulary from the probability distribution.
        """
        if self._vocab is None:
            self._vocab = set()
            for next_chars in self.prob_distribution.values():
                self._vocab.update(next_chars.keys())
        return self._vocab

    def get_ranges(self, context: Tuple[int, ...]) -> Dict[int, Tuple[int, int]]:
        """
        Compute cumulative integer ranges for each symbol under the given context.
        """
        if context in self._ranges_cache:
            return self._ranges_cache[context]

        # Fallback to uniform distribution if context is unseen
        if context not in self.prob_distribution:
            vocab = self.vocab
            prob_dict = {char: 1.0 / len(vocab) for char in vocab}
        else:
            prob_dict = self.prob_distribution[context]

        # Cache sorted symbols for consistent ordering
        if context not in self._sorted_symbols_cache:
            self._sorted_symbols_cache[context] = sorted(prob_dict.keys())
        sorted_symbols = self._sorted_symbols_cache[context]

        total_prob = sum(prob_dict.values())
        SCALE_FACTOR = min(10000, int((self.TOP_VALUE + 1) / 4))

        ranges = {}
        cumsum = 0

        for symbol in sorted_symbols:
            prob = prob_dict[symbol] / total_prob
            count = max(1, int(prob * SCALE_FACTOR))
            low = cumsum
            high = cumsum + count
            ranges[symbol] = (low, high)
            cumsum += count

        ranges['__total_count__'] = cumsum
        self._ranges_cache[context] = ranges
        return ranges

    def encode(self, message: List[int], debug=False) -> bytearray:
        """
        Encode a sequence of symbols into arithmetic-coded bit sequence.
        """
        output_bits = bytearray()
        bits_to_follow = 0

        low = 0
        high = self.TOP_VALUE
        context = tuple([self.start_token] * (self.n - 1))
        message_with_eom = message + [self.end_token]

        for i, symbol in enumerate(message_with_eom):
            if debug and i < 5:
                print(f"[Encode] Symbol {i}: {symbol}, low={low}, high={high}")

            ranges = self.get_ranges(context)
            total_count = ranges['__total_count__']

            if symbol not in ranges:
                raise ValueError(f"Symbol {symbol} not in vocabulary")

            symbol_low, symbol_high = ranges[symbol]
            range_width = high - low + 1

            high = low + int(range_width * symbol_high / total_count) - 1
            low = low + int(range_width * symbol_low / total_count)

            # Normalization loop
            while True:
                if high < self.HALF:
                    output_bits.append(0)
                    output_bits.extend([1] * bits_to_follow)
                    bits_to_follow = 0
                elif low >= self.HALF:
                    output_bits.append(1)
                    output_bits.extend([0] * bits_to_follow)
                    bits_to_follow = 0
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    bits_to_follow += 1
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                else:
                    break

                low = 2 * low
                high = 2 * high + 1

            if self.n == 1:
                context = ()
            else:
                context = context[1:] + (symbol,)

        # Final bit flushing
        bits_to_follow += 1
        if low < self.FIRST_QUARTER:
            output_bits.append(0)
            output_bits.extend([1] * bits_to_follow)
        else:
            output_bits.append(1)
            output_bits.extend([0] * bits_to_follow)

        return output_bits

    def decode(self, encoded_bits: bytearray, debug=False) -> List[int]:
        """
        Decode an arithmetic-coded bit sequence back into a symbol list.
        """
        if not encoded_bits:
            return []

        decoded_message = []

        low = 0
        high = self.TOP_VALUE
        value = 0

        bits_to_read = min(self.bits, len(encoded_bits))
        for i in range(bits_to_read):
            value = (value << 1) | encoded_bits[i]
        if bits_to_read < self.bits:
            value <<= (self.bits - bits_to_read)

        bit_index = self.bits
        context = tuple([self.start_token] * (self.n - 1))
        iteration = 0

        while True:
            iteration += 1
            ranges = self.get_ranges(context)
            total_count = ranges['__total_count__']
            range_width = high - low + 1
            scaled_value = ((value - low + 1) * total_count - 1) // range_width

            if context not in self._sorted_symbols_cache:
                self._sorted_symbols_cache[context] = sorted(
                    k for k in ranges.keys() if k != '__total_count__'
                )
            sorted_symbols = self._sorted_symbols_cache[context]

            symbol = None
            for sym in sorted_symbols:
                sym_low, sym_high = ranges[sym]
                if sym_low <= scaled_value < sym_high:
                    symbol = sym
                    break

            if symbol is None:
                if debug:
                    print(f"[Decode] Could not find symbol at iteration {iteration}")
                break

            if symbol == self.end_token:
                break

            decoded_message.append(symbol)

            if debug and iteration <= 5:
                print(f"[Decode] Iteration {iteration}: symbol={symbol}, low={low}, high={high}, value={value}")

            symbol_low, symbol_high = ranges[symbol]
            high = low + int(range_width * symbol_high / total_count) - 1
            low = low + int(range_width * symbol_low / total_count)

            while True:
                if high < self.HALF:
                    pass
                elif low >= self.HALF:
                    low -= self.HALF
                    high -= self.HALF
                    value -= self.HALF
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                    value -= self.FIRST_QUARTER
                else:
                    break

                low = 2 * low
                high = 2 * high + 1
                value = 2 * value
                if bit_index < len(encoded_bits):
                    value |= encoded_bits[bit_index]
                    bit_index += 1

            if self.n == 1:
                context = ()
            else:
                context = context[1:] + (symbol,)

        return decoded_message

    def clear_cache(self):
        """
        Clear all cached probability and range data.
        """
        self._ranges_cache.clear()
        self._sorted_symbols_cache.clear()





# =================== Profiling ===================

def _format_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _cpu_delta_s(before, after) -> tuple:
    user = after.user - before.user
    system = after.system - before.system
    return user, system, user + system


def profile_encode_decode(encoder, message: List[int]) -> dict:
    """Profile encode/decode: wall time (perf_counter), CPU time and RSS (psutil)."""
    process = psutil.Process()

    mem_before = process.memory_info().rss
    cpu_before = process.cpu_times()

    t0 = time.perf_counter()
    encoded = encoder.encode(message)
    encode_wall_s = time.perf_counter() - t0
    cpu_after_encode = process.cpu_times()
    mem_after_encode = process.memory_info().rss

    t0 = time.perf_counter()
    decoded = encoder.decode(encoded)
    decode_wall_s = time.perf_counter() - t0
    cpu_after_decode = process.cpu_times()
    mem_after_decode = process.memory_info().rss

    enc_user, enc_sys, enc_cpu = _cpu_delta_s(cpu_before, cpu_after_encode)
    dec_user, dec_sys, dec_cpu = _cpu_delta_s(cpu_after_encode, cpu_after_decode)

    return {
        "encoded": encoded,
        "decoded": decoded,
        "encode_wall_s": encode_wall_s,
        "decode_wall_s": decode_wall_s,
        "total_wall_s": encode_wall_s + decode_wall_s,
        "encode_cpu_user_s": enc_user,
        "encode_cpu_system_s": enc_sys,
        "encode_cpu_total_s": enc_cpu,
        "decode_cpu_user_s": dec_user,
        "decode_cpu_system_s": dec_sys,
        "decode_cpu_total_s": dec_cpu,
        "total_cpu_s": enc_cpu + dec_cpu,
        "mem_before_bytes": mem_before,
        "mem_after_encode_bytes": mem_after_encode,
        "mem_after_decode_bytes": mem_after_decode,
        "mem_delta_bytes": mem_after_decode - mem_before,
    }


def profile_stats_lines(stats: dict) -> List[str]:
    return [
        f"Wall time:  encode {stats['encode_wall_s']*1000:.4f} ms, "
        f"decode {stats['decode_wall_s']*1000:.4f} ms, "
        f"total {stats['total_wall_s']*1000:.4f} ms",
        f"CPU time:   encode {stats['encode_cpu_total_s']*1000:.4f} ms "
        f"(user {stats['encode_cpu_user_s']*1000:.4f}, "
        f"sys {stats['encode_cpu_system_s']*1000:.4f}), "
        f"decode {stats['decode_cpu_total_s']*1000:.4f} ms "
        f"(user {stats['decode_cpu_user_s']*1000:.4f}, "
        f"sys {stats['decode_cpu_system_s']*1000:.4f}), "
        f"total {stats['total_cpu_s']*1000:.4f} ms",
        f"Memory RSS: before {_format_bytes(stats['mem_before_bytes'])}, "
        f"after encode {_format_bytes(stats['mem_after_encode_bytes'])}, "
        f"after decode {_format_bytes(stats['mem_after_decode_bytes'])}, "
        f"delta {_format_bytes(stats['mem_delta_bytes'])}",
    ]


def print_profile_stats(stats: dict) -> None:
    for line in profile_stats_lines(stats):
        print(f"  {line}")


# =================== Test Function ===================

def test_encoder():
    """Run several test cases to validate encoder-decoder consistency."""
    print("=" * 60)
    print("Testing ArithmeticEncoder with simple N-gram model")
    print("=" * 60)

    prob_distribution = {
        (): {
            0: 0.5,
            1: 0.3,
            2: 0.2,
            -2: 0.0001  # End-of-message token
        }
    }

    encoder = ArithmeticEncoder(
        prob_distribution=prob_distribution,
        bits=32,
        start_token=-1,
        end_token=-2
    )

    test_cases = [
        [0, 1, 2],
        [0, 1, 2, 0, 1],
        [0] * 10,
        [1, 2, 0, 1, 2] * 5,
        [0, 1, 2] * 20,
    ]

    for i, test_seq in enumerate(test_cases):
        print(f"\nTest case {i + 1}: length={len(test_seq)}")
        print(f"Sequence: {test_seq[:20]}{'...' if len(test_seq) > 20 else ''}")

        try:
            stats = profile_encode_decode(encoder, test_seq)
            encoded = stats["encoded"]
            decoded = stats["decoded"]

            success = (decoded == test_seq)
            print(f"  Encoded bits: {len(encoded)}")
            print(f"  Decoded length: {len(decoded)}")
            print_profile_stats(stats)
            print(f"  Status: {'✓ SUCCESS' if success else '✗ FAILED'}")

            if not success:
                print(f"  Expected: {test_seq}")
                print(f"  Got:      {decoded}")
                break
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            break

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_encoder()