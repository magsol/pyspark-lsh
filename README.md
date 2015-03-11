Spark-LSH
=========

Locality-sensitive hashing for [Apache Spark](http://spark.apache.org/). Largely a PySpark port of the [spark-hash project](https://github.com/mrsqueeze/spark-hash).

Prerequisites
-------------

- Spark 1.2+
- Python 2.7+
- SciPy 0.15+
- NumPy 1.9+

Implementation Details
----------------------

This project follows the main workflow of the spark-hash Scala LSH implementation. Its core `lsh.py` module accepts an RDD-backed list of either dense NumPy arrays or PySpark SparseVectors, and generates a model that is simply a wrapper around all the intermediate RDDs generated. More details on each of these steps will follow.

It is important to note that while this pipeline will accept either dense or sparse vectors, the original hash function from [spark-hash](https://github.com/mrsqueeze/spark-hash) will almost certainly fail with dense vectors, resulting in all vectors being hashed into all bands. Work is currently underway to implement alternative hash functions that more evenly split dense vectors. For the sparse case, the results duplicate those of [spark-hash](https://github.com/mrsqueeze/spark-hash).

Usage
-----

Usage follows that of the spark-hash project. Parameters remain the same.

### Parameters

Command line-parameters:

 - *--bins [-m]*: Number of bins in which to hash the data. A smaller number of bins will increase the number of collisions, producing larger clusters.
 - *--numrows [-n]*: Number of times to hash the individual elements. A larger number will diversify the signatures, increasing the likelihood that similar elements will be hashed together. Put another way, this is the number of bits in the signatures.
 - *--bands [-b]*: Number of bands in which to split the signatures. Each band will have (n / b) elements. A smaller number of bands will increase the confidence in element similiarity.
 - *--minbucketsize [-c]*: Minimum allowable bucket size. Any buckets with fewer than this many elements will be dropped entirely.

Other parameters:

 - *p*: For use in the `minhash` hashing function. It is an integer that is larger than the number of dimensions of the original data, and is used to generate the random numbers that seed the minhash function.

### Tuning

As described in the MMDS book, LSH can be tuned by adjusting the number of rows and bands such that:

    threshold = (1.0 / bands) ** (1.0 / (rows / bands))
    
Naturally, the number of rows, bands, and the resulting size of the band (rows/bands) dictates the quality of results yielded by LSH. Higher thresholds produces clusters with higher similarity. Lower thresholds typically produce more clusters but sacrifices similarity. 

Regardless of parameters, it may be good to independently verify each cluster. One such verification method is to calculate the jaccard similarity of the cluster (it is a set of sets). [SciPy jaccard similarity](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html) is used, although future development will allow for user-supplied distance functions.

Future
------

Work on this project is ongoing and includes:

 - **Additional hashing functions** in particular to address the propensity for dense vectors to generate signatures of all 0s using `minhash`.
     - **Density sensitive hashing**, as outlined in the [Lin *et al* IEEE 2012 paper](http://arxiv.org/pdf/1205.2930.pdf).
     - **Dimension spans**, as outlined in the [Hefeeda *et al* HPDC 2012 paper](http://dl.acm.org/citation.cfm?id=2287111).
 - **Connection to distributed affinity matrix computation** as the initial groundwork for spectral clustering on Spark.