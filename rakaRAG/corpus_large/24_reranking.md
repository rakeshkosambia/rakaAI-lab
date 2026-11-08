# Reranking

A reranker looks at the question together with each candidate chunk and scores how well they fit. The usual pattern is to retrieve a wide set of candidates cheaply, with keyword scoring or vectors, then let the reranker keep the best few. Reranking improves the order of the results, not which documents were found, so a document missing from the candidates cannot be rescued. A reranker is slower and costs more than a plain search, so it is applied to a short list only. Graph based retrieval often produces a large candidate pool, which makes a reranker especially useful.
