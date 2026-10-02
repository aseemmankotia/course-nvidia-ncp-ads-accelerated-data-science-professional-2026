## Chapter 9 Cheat Sheet — cuGraph & cuSpatial

### cuGraph Algorithm Selection
| Need | Algorithm | Returns |
|---|---|---|
| Rank importance/influence | `cugraph.pagerank(G)` | df: vertex, pagerank |
| Find bridge/broker nodes | `cugraph.betweenness_centrality(G, k=...)` | df: vertex, betweenness_centrality |
| Find central-by-distance nodes | `cugraph.closeness_centrality(G)` | df: vertex, closeness_centrality |
| Find dense clusters/communities | `cugraph.louvain(G)` | (partition df, modularity_score) |

### Building a Graph
```python
import cugraph
G = cugraph.Graph(directed=True)  # or False for undirected
G.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='amount')
```

### Fraud-Ring Mapping
| Question | Algorithm |
|---|---|
| "Which accounts form a tight cluster/cycle?" | Louvain |
| "Which account bridges two rings?" | Betweenness centrality |
| "Which account is most globally influential?" | PageRank |
| "Is this a tabular classification problem (labeled rows, no relational structure)?" | cuML (RandomForest, XGBoost, etc.), not cuGraph |

### cuGraph vs cuML — Decision Rule
- **Data defined by connections/edges between entities** → cuGraph (pagerank, centrality, louvain, connected components).
- **Data defined by independent feature rows, labeled or unlabeled** → cuML (classification, regression, KMeans, DBSCAN).
- Same dataset can sometimes support both — the exam scenario tells you whether the *ask* is relational ("find clusters/rings/paths") or tabular ("classify/predict a value per row").

### cuSpatial — Awareness-Level Operations
| Operation | Purpose |
|---|---|
| Point-in-polygon | Test if coordinate(s) fall inside a defined boundary/zone |
| Nearest-neighbor query | Find closest point(s)/facility to each query point |
| Spatial join | Associate points with overlapping/containing polygons at scale |

### Key Parameters to Remember
- `pagerank`: default `alpha=0.85` (damping factor).
- `betweenness_centrality`: use `k=<sample size>` to approximate on large graphs — exact computation is expensive as edges/vertices grow.
- `louvain`: returns modularity_score to gauge community-detection quality (higher = better-defined communities).

### GPU Memory Considerations
- Graph memory scales primarily with **edge count** (COO/CSR structures), not vertex count.
- Very large graphs (hundreds of millions to billions of edges) may exceed single-GPU memory → consider **multi-GPU / dask-cugraph** distributed graph processing instead of a single `cugraph.Graph`.

### Common Exam Traps
- Confusing PageRank (importance) with Louvain (clustering) — they answer different questions.
- Using cuML KMeans/DBSCAN on transaction data when the scenario actually describes a *relational* fraud-ring pattern — wrong tool.
- Forgetting `k` sampling exists for betweenness_centrality and assuming exact computation is always feasible.
- Assuming cuSpatial handles arbitrary graph relationships — it's strictly geometric (containment, distance), not relational.
