import argparse
import numpy as np
import scipy.spatial.distance as distance

from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import SparseVector

import lsh
import hasher

def signatures(kv):
    """
    Generates the minhash signatures for each vector. Floods the network with
    <k, v> pairs of the form:

    <(vector id, band id), minhash>

    where each minhash is but a single number. The reason for this is to make
    it easier to aggregate the minhashes by their band index.
    """
    v, idx = kv
    hash_inits = _HASHES_.value
    m, n, b, _, p = _PARAMS_.value
    return [[(idx, i % b), hasher.minhash(v, h[0], h[1], p, m)] for i, h in enumerate(hash_inits)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Spark LSH',
        epilog = 'lol lsh', add_help = 'How to use',
        prog = 'python lsh.py <arguments>')
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
    hashes = np.vstack([np.random.random_integers(p, size = n), np.random.random_integers(0, p, size = n)]).T

    # Broadcast.
    _PARAMS_ = sc.broadcast([m, n, b, c, p])
    _HASHES_ = sc.broadcast(hashes)

    # Start by generating the signatures for each data point.
    # Output format is:
    # <(vector idx, band idx), minhash>
    signatures = zdata.flatMap(signatures).cache()

    # Put together the vector minhashes in the same band.
    # Output format is:
    # <(band idx, minhash list), vector idx>
    bands = signatures.groupByKey() \
        .map(lambda x: [(x[0][1], hash(x[1])), x[0][0]]) \
        .groupByKey().cache()

    # Should we filter?
    if c > 0:
        bands = bands.filter(lambda x: len(x[1]) > c).cache()

    # Remaps each element to a cluster / bucket index.
    # Output format is:
    # <vector idx, bucket idx>
    vector_bucket = bands.map(lambda x: frozenset(sorted(x[1]))).distinct() \
        .zipWithIndex().flatMap(lambda x: map(lambda y: (np.long(y), x[1]), x[0])) \
        .cache()

    # Reverses indices, to key the vectors by their buckets.
    # Output format is:
    # <bucket idx, vector idx>
    bucket_vector = vector_bucket.map(lambda x: (x[1], x[0])).cache()

    # Joins indices up with original data to provide clustering results.
    # Output format is:
    # <bucket idx, list of vectors>
    buckets = zdata.map(lambda x: (x[1], x[0])).join(vector_bucket) \
        .map(lambda x: (x[1][1], x[1][0])).groupByKey().cache()

    # Computes Jaccard similarity of each bucket.
    scores = buckets.map(lambda x: (x[0], distance.pdist(np.array(x[1].zdata), 'jaccard').sum())).cache()
