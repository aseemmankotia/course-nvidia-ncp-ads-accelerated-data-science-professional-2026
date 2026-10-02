## Chapter 9: Graph Analytics with cuGraph and Geospatial Awareness with cuSpatial — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | cugraph.pagerank default damping factor (alpha)? | 0.85. Exam tip: matches the classic PageRank paper default — don't confuse with betweenness_centrality's k sampling parameter. |
| 2 | What does the `k` parameter control in cugraph.betweenness_centrality? | Number of source vertices sampled to approximate the score on large graphs, trading accuracy for speed. Omit k for exact (slow) computation. |
| 3 | What does cugraph.louvain() return? | A tuple: (partition dataframe with vertex→community assignment, modularity_score float). Exam tip: modularity score indicates community-detection quality. |
| 4 | Correct method to load a cuDF edge-list dataframe into a cugraph.Graph? | G = cugraph.Graph(directed=True\|False); G.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='weight') |
| 5 | Fraud ring detection: which cuGraph algorithm finds the clusters, and which finds the bridge nodes between clusters? | Louvain (community detection) finds the ring clusters; betweenness_centrality finds bridge/broker accounts connecting rings. |
| 6 | cuGraph vs cuML — when do you choose each for a described analytics problem? | cuGraph: problem is defined by relationships/edges between entities (fraud rings, social networks, routing). cuML: problem is defined by independent rows of tabular features (classification, regression, clustering on attributes). |
| 7 | cuSpatial operation to test whether GPS points fall inside defined zone/region boundaries? | Point-in-polygon query. Exam tip: this is the core 'awareness-level' geospatial operation tested, distinct from graph relational algorithms. |
| 8 | cuSpatial operation to find the closest facility (e.g., warehouse) to each of many query points? | Nearest-neighbor query. Exam tip: don't substitute cuML's tabular NearestNeighbors — cuSpatial's is geometry-aware and GPU-optimized for coordinates. |
| 9 | PageRank vs betweenness_centrality vs Louvain — what does each measure? | PageRank: vertex importance/influence via incoming links. Betweenness centrality: how often a vertex lies on shortest paths (bridge/broker role). Louvain: partitions graph into densely-connected communities. |
| 10 | What dominates GPU memory usage when building a cugraph.Graph from a large edge list? | Edge count (COO/CSR storage), not vertex count. Exam tip: billion-edge graphs may require multi-GPU distributed cuGraph (dask-cugraph) rather than a single-GPU Graph object. |
| 11 | cugraph.pagerank() output dataframe columns? | 'vertex' and 'pagerank'. Exam tip: sort descending on 'pagerank' to find the most influential/central node in a fraud graph. |
| 12 | Why is exact betweenness_centrality often impractical on large transaction graphs? | Its complexity scales with vertex and edge counts across all shortest-path computations, making it expensive on graphs with millions of edges. Use the k sampling parameter to approximate. |
