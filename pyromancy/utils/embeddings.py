import numpy as np
from tqdm import tqdm


def load_word_vectors(filename):
    word_to_index = {}
    word_vectors = []

    with open(filename) as fp:
        for line in tqdm(fp.readlines()):
            line = line.split(" ")

            word = line[0]
            word_to_index[word] = len(word_to_index)

            vec = np.array([float(x) for x in line[1:]])
            word_vectors.append(vec)
    word_vector_size = len(word_vectors[0])
    return word_to_index, word_vectors, word_vector_size
