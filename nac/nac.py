from ngram import NGramModel
from arithmetic_coding import ArithmeticEncoder
import sys, logging, os, time


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

for i in range(len(codes)):
    logger.info(f"Code: {i}")
    test_sequence = codes[i]

    t0 = time.perf_counter()
    encoded_bits = encoder.encode(test_sequence)
    t1 = time.perf_counter()
    encode_ms = (t1 - t0) * 1000.0
    encode_times_ms.append(encode_ms)

    logger.info(f"Encoded: {len(encoded_bits)} bits, encode_time={encode_ms:.4f} ms")
    rate = len(encoded_bits) / (len(test_sequence) * 11)
    avgrate.append(rate)
    logger.info(f"Compression Rate: {rate:.2%}")

    t0 = time.perf_counter()
    decoded_sequence = encoder.decode(encoded_bits)
    t1 = time.perf_counter()
    decode_ms = (t1 - t0) * 1000.0
    decode_times_ms.append(decode_ms)

    logger.info(f"Decode_time={decode_ms:.4f} ms")
    logger.info(f"Verification: {'Correct' if decoded_sequence == test_sequence else 'Wrong'}")

logger.info(f"Average Compression Rate: {sum(avgrate)/len(avgrate):.2%}")
logger.info("")
logger.info("Timing summary (ms):")
avg_encode = sum(encode_times_ms) / len(encode_times_ms)
avg_decode = sum(decode_times_ms) / len(decode_times_ms)
logger.info(f"  Average encode time: {avg_encode:.4f} ms")
logger.info(f"  Average decode time: {avg_decode:.4f} ms")
logger.info(f"  encode: min={min(encode_times_ms):.4f}, max={max(encode_times_ms):.4f}")
logger.info(f"  decode: min={min(decode_times_ms):.4f}, max={max(decode_times_ms):.4f}")




