import argparse
import numpy as np

from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import SparseVector

import lsh

def read_text(sc, path):
    text = sc.textFile(path).collect()
    data = []
    for s in text:
        elements = map(int, s[s.index("[") + 1:s.index("]")].split(","))
        v = SparseVector(65535, elements, np.ones(len(elements)))
        data.append(v)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Spark LSH',
        epilog = 'lol lsh', add_help = 'How to use',
        prog = 'python driver.py <arguments>')
    parser.add_argument("-i", "--input", required = True,
        help = "Input directory of text files.")

    # Optional parameters.
    parser.add_argument("-m", "--bins", type = int, default = 1000,
        help = "Number of bins into which to hash the data. Smaller numbers " +
            "increase collisions, producing larger clusters. [DEFAULT: 1000]")
    parser.add_argument("-n", "--numrows", type = int, default = 1000,
        help = "Number of times to hash the elements. Larger numbers diversify " +
            "signatures, increasing likelihood similar vectors will be hashed together. " +
            "This is also the length of the signature. [DEFAULT: 1000]")
    parser.add_argument("-b", "--bands", type = int, default = 25,
        help = "Number of bands. Each band will have (n / b) elements. Larger " +
            "numbers of elements increase confidence in element similarity. [DEFAULT: 25]")
    parser.add_argument("-c", "--minbucketsize", type = int, default = 2,
        help = "Minimum bucket size (0 to disable). Buckets with fewer than this " +
            "number of elements will be dropped. [DEFAULT: 2]")

    args = vars(parser.parse_args())
    sc = SparkContext(conf = SparkConf())

    # Read the input data.
    data = read_text(sc, args['input'])
    p = 65537
    m, n, b, c = args['bins'], args['numrows'], args['bands'], args['minbucketsize']
    model = lsh.run(data, p, m, n, b, c)
    print 'Found %s clusters.' % model.buckets.count()
