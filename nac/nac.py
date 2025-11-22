from ngram import NGramModel
from arithmetic_coding import ArithmeticEncoder
import sys, logging, os


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

for i in range(len(codes)):
    logger.info(f"Code: {i}")
    test_sequence = codes[i]
    encoded_bits = encoder.encode(test_sequence)
    # print(f"\nOriginal Sequence: {test_sequence}")
    # print(encoded_bits)
    logger.info(f"Encoded: {len(encoded_bits)} bits")
    rate = len(encoded_bits) / (len(test_sequence) * 11)
    avgrate.append(rate)
    logger.info(f"Compression Rate: {rate:.2%}")


    decoded_sequence = encoder.decode(encoded_bits)
    # print(f"Decoded Sequence: {decoded_sequence}")
    logger.info(f"Verification: {'Correct' if decoded_sequence == test_sequence else 'Wrong'}")

logger.info(f"Average Compression Rate: {sum(avgrate)/len(avgrate):.2%}")




