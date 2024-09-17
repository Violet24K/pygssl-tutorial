PyG-SSL Datasets
=====================
Before using PyG-SSL, you need to convert your dataset into :class:`torch_geometric.data.Dataset`, as we use the data structure and loader from torch_geometric for accelerated computation. 




Augment
-------------------
PyG-SSL provides many standard augmentations to use for creating positive and negative examples. 

.. class:: RandomDropNode(is_x: bool = True, is_adj: bool = False, drop_percent=0.2)

	Randomly drop a percentage of nodes or edges from the graph.

	Parameters:
	-----------
	- **is_x** (bool, optional):
		Whether to drop nodes from the feature matrix. Default is True.

	- **is_adj** (bool, optional):
		Whether to drop edges from the adjacency matrix. Default is False.

	- **drop_percent** (float, optional):
		The percentage of nodes or edges to drop. Default is 0.2.


	

.. class:: RandomDropEdge(is_x: bool = True, is_adj: bool = False, drop_percent=0.2)

	Randomly drop a percentage of edges or edges from the graph.

	Parameters:
	-----------
	- **is_x** (bool, optional):
		Whether to drop nodes from the feature matrix. Default is True.

	- **is_adj** (bool, optional):
		Whether to drop edges from the adjacency matrix. Default is False.

	- **drop_percent** (float, optional):
		The percentage of nodes or edges to drop. Default is 0.2.



.. class:: RandomMask(is_x: bool = True, is_adj: bool = False, drop_percent=0.2)

	Randomly mask a percentage of nodes features from the graph.

	Parameters:
	-----------
	- **is_x** (bool, optional):
		Whether to drop nodes from the feature matrix. Default is True.

	- **is_adj** (bool, optional):
		Whether to drop edges from the adjacency matrix. Default is False.

	- **drop_percent** (float, optional):
		The percentage of nodes or edges to drop. Default is 0.2.


.. class:: RandomMaskChannel(is_x: bool = True, is_adj: bool = False, drop_percent=0.2)

	Similar to RandomMask, but mask a percentage of feature channels of the nodes.

	Parameters:
	-----------
	- **is_x** (bool, optional):
		Whether to drop nodes from the feature matrix. Default is True.

	- **is_adj** (bool, optional):
		Whether to drop edges from the adjacency matrix. Default is False.

	- **drop_percent** (float, optional):
		The percentage of nodes or edges to drop. Default is 0.2.




.. class:: ShuffleNode()
	
	Shuffle the nodes in the graph.

	Parameters:
	-----------
	- **None**



.. class:: AugmentSubgraph(is_x: bool = True, is_adj: bool = False, drop_percent=0.2)

	Randomly drop a percentage subgraph from the graph.

	Parameters:
	-----------
	- **is_x** (bool, optional):
		Whether to drop nodes from the feature matrix. Default is True.

	- **is_adj** (bool, optional):
		Whether to drop edges from the adjacency matrix. Default is False.

	- **drop_percent** (float, optional):
		The percentage of nodes or edges to drop. Default is 0.2.


.. class:: NeighborSearch_AFGRL(device="cuda", num_centroids=100, num_kmeans=5, clus_num_iters=20)

	Neighbor search for AFGRL.

	Parameters:
	-----------
	- **device** (str, optional):
		Device to use for computation. Default is "cuda".

	- **num_centroids** (int, optional):
		Number of centroids to use for clustering. Default is 100.

	- **num_kmeans** (int, optional):
		Number of k-means to use for clustering. Default is 5.

	- **clus_num_iters** (int, optional):
		Number of iterations for clustering. Default is 20.


.. class:: SumEmb(pooler: str="avg", act: Optional[Callable]=None)

	Get the summary for a batch of embeddings.

	Parameters:
	-----------
	- **pooler** (str, optional):
		"avg" or "max", specifies how to generate global embeddings. Default is "avg".

	- **act** (Callable, optional):
		Activation function for the encoder. Default is None.




Transform
-------------------

.. class:: TransformList(transform_list: List[Callable])

	A list of transform function

	Parameters:
	-----------
	- **transform_list** (List[Callable]):
		A list of transform functions to apply.



.. class:: Edge2Adj(norm: Optional[Callable] = None)

	Convert edge to adjacency matrix.

	Parameters:
	-----------
	- **norm** (Optional[Callable]): Normalization of the adjacency matrix.
			(default: None) None: Generate 0/1 adjacency matrix from edge_index & edge_attr/edge_weight.



.. class:: GCNNorm(add_self_loops: int = 1)

	Applies the GCN normalization from the `"Semi-supervised Classification with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_ paper (functional name: :obj:`gcn_norm`).

	.. math::
		\mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
		\mathbf{\hat{D}}^{-1/2}

	where :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij} + 1`.

	Parameters:
	-----------
	- **add_self_loops** (int, optional):
		Number of self loops to add to the adjacency matrix. Default is 1.

.. class:: NormalizeFeatures(attrs: List[str] = ["x"], ord: Optional[Any] = None)
	
	Row-normalizes the attributes given in :obj:`attrs` to sum-up to one (functional name: :obj:`normalize_features`).

	Parameters:
	-----------
	- **attrs** (List[str]): The names of attributes to normalize.
			(default: :obj:`["x"]`)

	- **ord** (Optional[str, int, float]): The order of normalization.
			(default: "sum1")
			sum1: x = x - min(x), x / sum(x)
			other supported normalization:
			https://pytorch.org/docs/stable/generated/torch.linalg.norm.html#torch.linalg.norm