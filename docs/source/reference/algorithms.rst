PyG-SSL Algorithms
=====================

Methods
-------------------

Please refer to the "Tutorial" section and our provided jupiter notebooks for the detailed usage of the algorithms.



Neural Networks
-------------------

.. class:: MLPnum_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm")

	Multi-layer perceptron (MLP) model.

	Parameters:
	-----------
	- **num_layers** (int): 
		The number of layers.
	
	- **input_dim** (int):
		The input feature dimension.
	
	- **hidden_dim** (int):
		The hidden feature dimension.
	
	- **output_dim** (int):
		The output feature dimension.
	
	- **activation** (str):
		The activation function. Default is "relu".
	
	- **norm** (str):
		The type of normalization layer. Default is "batchnorm".



.. class:: NormLayer(hidden_dim, norm_type)

	Normalization layer for the neural networks.

	Parameters:
	-----------
	- **hidden_dim** (int): 
		The hidden dimension of the normalization layer.
	
	- **norm_type** (str):
		The type of normalization layer. Default is "batch".


.. class:: ApplyNodeFunc(mlp, norm="batchnorm", activation="relu")

	Apply a function to the node features.

	Parameters:
	-----------
	- **mlp** (torch.nn.Module): 
		The multi-layer perceptron.
	
	- **norm** (str):
		The type of normalization layer. Default is "batchnorm".
	
	- **activation** (str):
		The activation function. Default is "relu".



.. class:: GAT(in_dim, num_hidden, out_dim, num_layers, nhead, nhead_out, activation, feat_drop, attn_drop, negative_slope, residual, norm, concat_out=False, encoding=False)

	Graph Attention Networks (GAT) from the "Graph Attention Networks" paper.

	Parameters:
	-----------
	- **in_dim** (int): 
		The input feature dimension.
	
	- **num_hidden** (int):
		The hidden feature dimension.
	
	- **out_dim** (int):
		The output feature dimension.
	
	- **num_layers** (int):
		The number of layers.
	
	- **nhead** (int):
		The number of head attentions in layer.
	
	- **nhead_out** (int):
		The number of head attentions in the output layer.
	
	- **activation** (Callable):
		The activation function.
	
	- **feat_drop** (float):
		The dropout rate for feature.
	
	- **attn_drop** (float):
		The dropout rate for attention.
	
	- **negative_slope** (float):
		The negative slope of leaky ReLU.
	
	- **residual** (bool):
		Use residual connection or not.
	
	- **norm** (str):
		The type of normalization layer.
	
	- **concat_out** (bool):
		Concatenate the output of all heads or not.
	
	- **encoding** (bool):
		Use encoding mode or not.


.. class:: DotGAT(in_dim, num_hidden, out_dim, num_layers, nhead, nhead_out, activation, feat_drop, attn_drop, residual, norm, concat_out=False, encoding=False)

	The GAT model with dot-product attention.

	Parameters:
	-----------

	- **in_dim** (int): 
		The input feature dimension.
	
	- **num_hidden** (int):
		The hidden feature dimension.
	
	- **out_dim** (int):
		The output feature dimension.

	- **num_layers** (int):
		The number of layers.
	
	- **nhead** (int):
		The number of head attentions in layer.

	- **nhead_out** (int):
		The number of head attentions in the output layer.

	- **activation** (Callable):
		The activation function.

	- **feat_drop** (float):
		The dropout rate for feature.

	- **attn_drop** (float):
		The dropout rate for attention.
	
	- **residual** (bool):
		Use residual connection or not.
	
	- **norm** (str):
		The type of normalization layer.
	
	- **concat_out** (bool):
		Concatenate the output of all heads or not.

	- **encoding** (bool):
		Use encoding mode or not.

.. class:: GCN(in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, encoding=False)

	The Graph Convolutional Networks (GCN) model.

	Parameters:
	-----------

	- **in_dim** (int): 
		The input feature dimension.
	
	- **num_hidden** (int):
		The hidden feature dimension.
	
	- **out_dim** (int):
		The output feature dimension.
	
	- **num_layers** (int):
		The number of layers.
	
	- **dropout** (float):
		The dropout rate.
	
	- **activation** (Callable):
		The activation function.
	
	- **residual** (bool):
		Use residual connection or not.
	
	- **norm** (str):
		The type of normalization layer.
	
	- **encoding** (bool):
		Use encoding mode or not.

	
.. class:: GIN(in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, encoding=False, learn_eps=False, aggr="sum")

	The Graph Isomorphism Network (GIN) model.

	Parameters:
	-----------

	- **in_dim** (int): 
		The input feature dimension.
	
	- **num_hidden** (int):
		The hidden feature dimension.
	
	- **out_dim** (int):
		The output feature dimension.
	
	- **num_layers** (int):
		The number of layers.
	
	- **dropout** (float):
		The dropout rate.
	
	- **activation** (Callable):
		The activation function.
	
	- **residual** (bool):
		Use residual connection or not.
	
	- **norm** (str):
		The type of normalization layer.
	
	- **encoding** (bool):
		Use encoding mode or not.
	
	- **learn_eps** (bool):
		Learn the epsilon parameter or not.
	
	- **aggr** (str):
		The type of aggregation function.


