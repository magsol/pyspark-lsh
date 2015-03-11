import hasher
import functools
import numpy as np
import scipy.spatial.distance as distance

from pyspark.mllib.linalg import SparseVector

def distance_metric(kv):
    """
    Generates a pairwise, summed Jaccard distance metric for all the elements
    in a cluster / bucket. Input <k, v> pairs are of the form:

    <bucket id, list of vectors>

    Output <k, v> pair is: <bucket id, jaccard distance>
    """
    bid, X = kv[0], kv[1].data
    if type(X[0]) is SparseVector:
        Y = np.array([x.toArray() for x in X])
        X = Y
    return [bid, distance.pdist(np.array(X), 'jaccard').sum()]

def run(data, p, m, n, b, c):
    """
    Starts the main LSH process.

    Parameters
    ----------
    zdata : RDD[Vector]
        RDD of data points. Acceptable vector types are numpy.ndarray
        or PySpark SparseVector.
    p : integer, larger than the largest value in data.
    m : integer, number of bins for hashing.
    n : integer, number of rows to split the signatures into.
    b : integer, number of bands.
    c : integer, minimum allowable cluster size.
    """
    zdata = data.zipWithIndex()
    seeds = np.vstack([np.random.random_integers(p, size = n), np.random.random_integers(0, p, size = n)]).T
    hashes = [functools.partial(hasher.minhash, a = s[0], b = s[1], p = p, m = m) for s in seeds]

    # Start by generating the signatures for each data point.
    # Output format is:
    # <(vector idx, band idx), minhash>
    sigs = zdata.flatMap(lambda x: [[(x[1], i % b), hashes[i](x[0])] for i, h in enumerate(hashes)]).cache()

    # Put together the vector minhashes in the same band.
    # Output format is:
    # <(band idx, minhash list), vector idx>
    bands = sigs.groupByKey() \
        .map(lambda x: [(x[0][1], hash(frozenset(x[1].data))), x[0][0]]) \
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
    scores = buckets.map(distance_metric).cache()

    # Return a wrapper object around the metrics of interest.
    return PyLSHModel(sigs, bands, vector_bucket, bucket_vector, buckets, scores)

class PyLSHModel:

    def __init__(self, sigs, bands, v_b, b_v, buckets, scores):
        self.signatures = sigs
        self.bands = bands
        self.vectors_buckets = v_b
        self.buckets_vectors = b_v
        self.buckets = buckets
        self.scores = scores
