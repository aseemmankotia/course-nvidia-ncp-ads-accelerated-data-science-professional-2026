## Chapter 9: Graph Analytics with cuGraph and Geospatial Awareness with cuSpatial — Practice Questions

**Q1.** A payments company wants to detect tightly-knit clusters of accounts that repeatedly cycle money among themselves — a signature of coordinated fraud rings. Which cuGraph function BEST surfaces these clusters directly from the transaction graph?

A) cugraph.pagerank
B) cugraph.louvain
C) cuml.KMeans
D) cugraph.betweenness_centrality

<details>
<summary>Answer</summary>

**Correct: B**

Louvain is a community-detection algorithm that partitions a graph into densely-connected sub-groups (communities), which is exactly the structure a fraud ring exhibits.

**Why the others are wrong:**
- PageRank ranks vertex importance/influence, it does not partition the graph into groups.
- KMeans is a cuML tabular clustering algorithm operating on feature vectors — it ignores the relational edge structure that defines a ring.
- Betweenness centrality finds bridge/broker nodes between clusters, not the clusters themselves.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** Investigators have already found two separate fraud rings using community detection. They now need to identify the single account most likely acting as the go-between that launders funds from one ring to the other. Which cuGraph algorithm should they run?

A) cugraph.louvain
B) cugraph.pagerank
C) cugraph.jaccard
D) cugraph.betweenness_centrality

<details>
<summary>Answer</summary>

**Correct: D**

Betweenness centrality measures how often a vertex lies on shortest paths between other vertices, which is exactly the signature of a bridge/broker node connecting two clusters.

**Why the others are wrong:**
- Louvain would just re-detect the two existing communities, not the bridging node.
- PageRank measures overall importance/influence within the whole graph, not path-brokering between clusters.
- Jaccard similarity measures neighborhood overlap between vertex pairs, not global path centrality.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** A team has a table of individual transactions (amount, merchant category, timestamp, location) with a labeled fraud/not-fraud column and wants to train a classifier to score new transactions. Which library is the correct fit for this task?

A) cuGraph, using pagerank scores as the fraud label
B) cuML, using a supervised classifier such as RandomForestClassifier
C) cuGraph, by building a graph of transactions and running louvain
D) cuSpatial, using point-in-polygon on transaction locations

<details>
<summary>Answer</summary>

**Correct: B**

This is a row-wise, i.i.d. tabular supervised-learning problem with labeled features — the canonical cuML use case, not a relational/graph problem.

**Why the others are wrong:**
- PageRank produces an unsupervised importance score, not a trained classifier on labeled tabular data.
- Louvain finds communities in an edge list; there's no described relationship graph here, only per-row features.
- cuSpatial only handles geometric queries, not general classification of tabular fraud features.

</details>

---

**Q4.** A logistics platform needs to determine, for each of 5 million GPS-tagged delivery addresses, whether the point falls inside any of 200 defined city delivery-zone polygons. Which capability should be used?

A) cugraph.louvain on a graph of addresses
B) cuML DBSCAN clustering on lat/long
C) cuDF groupby on zone_id column
D) cuSpatial point-in-polygon query

<details>
<summary>Answer</summary>

**Correct: D**

Point-in-polygon is cuSpatial's core geospatial-awareness operation for testing whether points fall within polygon boundaries at scale on the GPU.

**Why the others are wrong:**
- Louvain is for relational community detection on edge lists, irrelevant to geometry.
- DBSCAN clusters points by density but cannot test exact polygon membership against defined zone boundaries.
- cuDF groupby requires the zone already assigned per row — it doesn't compute the spatial test itself.

</details>

---

**Q5.** A retailer wants to find, for each of 2 million customer coordinates, the closest warehouse from a set of 50 warehouse coordinates. Which cuSpatial capability directly supports this?

A) cuSpatial nearest-neighbor query
B) cuSpatial point-in-polygon
C) cuGraph betweenness_centrality
D) cuML NearestNeighbors on a CPU fallback

<details>
<summary>Answer</summary>

**Correct: A**

cuSpatial provides GPU-accelerated nearest-neighbor / distance queries designed for this exact point-to-point proximity use case at scale.

**Why the others are wrong:**
- Point-in-polygon tests containment within a boundary, not closest-point distance.
- Betweenness centrality is a graph-relational metric, unrelated to spatial coordinates.
- Falling back to a CPU-based cuML equivalent defeats the purpose of GPU-accelerated geospatial awareness and cuML doesn't provide a geospatial nearest-neighbor primitive for this.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** Running exact betweenness_centrality on a 50-million-edge transaction graph is too slow for the exam scenario's SLA. What should be changed in the call to get an approximate result faster?

A) Pass a smaller sample size via the k parameter
B) Set normalized=False
C) Switch to cugraph.pagerank instead
D) Increase max_iter

<details>
<summary>Answer</summary>

**Correct: A**

betweenness_centrality accepts a k parameter that samples a subset of source vertices to approximate the score, trading accuracy for speed on large graphs.

**Why the others are wrong:**
- normalized only rescales the score range, it doesn't change the compute cost.
- Switching algorithms changes what is measured entirely; it isn't a legitimate way to speed up the same metric.
- max_iter is not a parameter of betweenness_centrality; it's associated with iterative algorithms like pagerank.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** You have a cuDF dataframe `gdf` with columns 'src', 'dst', and 'amount' representing weighted directed transfers. What is the correct way to load this into a cuGraph Graph object?

A) G = cugraph.Graph(directed=True); G.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='amount')
B) cugraph.Graph(gdf, weight='amount')
C) G = cugraph.Graph(); G.load_edgelist(gdf)
D) cugraph.from_pandas_edgelist(gdf.to_pandas(), 'src', 'dst')

<details>
<summary>Answer</summary>

**Correct: A**

The correct pattern is instantiating cugraph.Graph(directed=True/False) and populating it via from_cudf_edgelist, specifying source, destination, and optionally edge_attr for weights.

**Why the others are wrong:**
- cugraph.Graph does not accept a dataframe directly in its constructor.
- load_edgelist is not the cuGraph API method name for this operation.
- Converting to pandas and using a networkx-style function defeats GPU acceleration and is not the cuGraph API.

</details>

---

**Q8.** Why is cuGraph the preferred choice over cuML for the fraud-ring detection scenario, even though both run on GPU?

A) cuGraph is always faster than cuML regardless of data shape
B) cuML cannot process categorical features required for fraud detection
C) cuGraph supports supervised learning while cuML only supports unsupervised learning
D) cuGraph algorithms operate natively on relationships between entities (edges), while cuML algorithms operate on independent tabular feature rows

<details>
<summary>Answer</summary>

**Correct: D**

The defining trait of a fraud ring is the connection pattern between accounts, not the standalone features of any single account — that structural, relational information is exactly what graph algorithms like Louvain and centrality are built to exploit.

**Why the others are wrong:**
- Speed depends on the algorithm and data; cuGraph isn't universally faster, it's structurally more appropriate.
- cuML fully supports categorical/encoded features; that's not the limiting factor here.
- Both libraries support supervised and unsupervised approaches; that isn't the differentiator being tested.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

