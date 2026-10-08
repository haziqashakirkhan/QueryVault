# Vector Databases

A vector database stores embeddings and answers one question very quickly: which stored vectors are nearest to this query vector? This operation is called nearest-neighbour search. Ordinary databases match exact values, while a vector database ranks results by closeness in meaning.

## ChromaDB

ChromaDB is an open-source vector database that runs inside the Python process and can persist data to disk. Data is organised into collections. Each record holds an id, the original document text, its embedding and optional metadata such as the file name or page number. By default ChromaDB measures distance with squared L2, also known as squared Euclidean distance. It also supports cosine distance and inner product, which can be selected when the collection is created. Metadata filters let a query search only some records, for example one source file.

## FAISS

FAISS is a library from Meta for fast similarity search over very large sets of vectors. It focuses on the index and the search itself, so it does not store the original text or metadata unless the developer builds that layer separately. FAISS is a strong choice for large-scale or GPU-accelerated search.

## Exact and approximate search

Exact search compares the query with every stored vector, which is accurate but slow for millions of records. Approximate nearest-neighbour methods such as the HNSW graph index trade a very small loss in accuracy for a large gain in speed. ChromaDB uses HNSW internally.
