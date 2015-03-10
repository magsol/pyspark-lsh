Spark-LSH
=========

Locality-sensitive hashing for [Apache Spark](http://spark.apache.org/). Largely a PySpark port of the [spark-hash project](https://github.com/mrsqueeze/spark-hash).

Prerequisites
-------------

- Python 2.7+
- SciPy 0.15+
- NumPy 1.9+

Implementation Details
----------------------

The project is currently in its infancy. This will be updated as aspects of the project come online.

Usage
-----

### Parameters

Will update.

### Tuning

As described in the MMDS book, LSH can be tuned by adjusting the number of rows and bands such that:

    threshold = (1.0 / bands) ** (1.0 / (rows / bands))
    
Naturally, the number of rows, bands, and the resulting size of the band (rows/bands) dictates the quality of results yielded by LSH. Higher thresholds produces clusters with higher similarity. Lower thresholds typically produce more clusters but sacrifices similarity. 

Regardless of parameters, it may be good to independently verify each cluster. One such verification method is to calculate the jaccard similarity of the cluster (it is a set of sets). [SciPy jaccard similarity](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html) is used.