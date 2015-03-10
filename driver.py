import argparse
import numpy as np

from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import SparseVector

import lsh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Spark LSH',
        epilog = 'lol lsh', add_help = 'How to use',
        prog = 'python driver.py <arguments>')
    parser.add_argument("-i", "--input", required = True,
        help = "Input directory of text files.")
    parser.add_argument("-p", "--primer", type = int, default = None,
        help = "An integer larger than the largest value in your data. This " +
            "primes the hashing functions and must be specified.")

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
    parser.add_argument("-c", "--minbucketsize", type = int, default = 0,
        help = "Minimum bucket size (0 to disable). Buckets with fewer than this " +
            "number of elements will be dropped. [DEFAULT: 0]")

    args = vars(parser.parse_args())
    sc = SparkContext(conf = SparkConf())

    # Read the input data and count the number of elements.
    #data = sc.textFile(args['input']).zipWithIndex()
    rawdata = np.random.binomial(1, 0.01, size = (275, 65535))
    data = [SparseVector(65535, np.where(row > 0)[0], np.ones(row[row > 0].shape[0])) for row in rawdata]
    zdata = sc.parallelize(data).zipWithIndex()
    p, m, n, b, c = args['primer'], args['bins'], args['numrows'], args['bands'], args['minbucketsize']
    print lsh.run(zdata, p, m, n, b, c)
