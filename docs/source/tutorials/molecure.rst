Molecular Graph Algorithms
===================================


InfoGraph
-------------------

InfoGraph is a groundbreaking framework designed for unsupervised and semi-supervised graph-level representation learning through the principle of mutual information maximization. As graph-structured data continues to proliferate across various domains—including social networks, biological systems, and knowledge graphs—the need for effective methods to learn meaningful representations has become increasingly vital. InfoGraph addresses this challenge by providing a robust approach that leverages mutual information to capture the intricate relationships within graph data.

Introduction
^^^^^^^^^^^^^^^^^^
The core concept of InfoGraph revolves around maximizing mutual information between different views of the same graph to learn rich, informative representations. Traditional graph representation learning methods often rely on node-level tasks or require extensive labeled data, limiting their applicability. InfoGraph, in contrast, offers a unified framework that can operate effectively in both unsupervised and semi-supervised settings, making it highly versatile.

Key Concepts
^^^^^^^^^^^^^^^^^^
1. **Mutual Information Maximization**: At the heart of InfoGraph is the idea of maximizing mutual information between representations derived from different graph views. By doing so, the framework encourages the model to learn features that are invariant to transformations, leading to robust and generalizable graph representations.

2. **Graph-Level Representation Learning**: InfoGraph shifts the focus from node-level embeddings to graph-level representations, enabling it to capture the overall structure and semantics of entire graphs. This is particularly beneficial in tasks where understanding the global properties of graphs is essential.

3. **Unsupervised and Semi-Supervised Learning**: InfoGraph is designed to work effectively in both unsupervised and semi-supervised scenarios. In unsupervised settings, it leverages unlabeled data to learn representations, while in semi-supervised settings, it can incorporate a limited amount of labeled data to enhance learning performance.

Methodology
^^^^^^^^^^^^^^^^^^
The methodology of InfoGraph consists of several key steps:

- **Graph Representation Generation**: The process begins by creating multiple views of the graph, which may involve various transformations or sub-sampling techniques. These views serve as the basis for learning diverse representations.

- **Mutual Information Objective**: InfoGraph employs a mutual information objective that encourages the model to maximize the information shared between the different views. This is achieved through carefully designed loss functions that facilitate the learning process.

- **Training and Optimization**: The model is trained using efficient optimization techniques that ensure convergence while effectively capturing the relationships within the graph. The focus on mutual information helps refine the learned representations iteratively.


API Reference in PyG-SSL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: Encoder(in_channels, hidden_channels, num_layers, GNN=GINConv)

	The Deep Graph Infomax Algorithm.

	Parameters:
	-----------
	- **in_channel** (int, optional):
  		Number of input features of the input dataset.

	- **hidden_channels** (int, optional): 
  		Number of hidden channels for the encoder.

	- **num_layers** (int, optional):
  		Number of layers for the encoder.

	- **GNN** (Optional[:class:`torch.nn.Module`]):
		The GNN to be used. Default is GINConv.


.. class:: InfoGraph(encoder: torch.nn.Module, hidden_channels: int, readout: Union[Callable, torch.nn.Module] = AvgReadout(), loss_function: Optional[torch.nn.Module] = LocalGlobalLoss(), gamma=.1, num_layers=1, prior=False)

	The Augmentation-Free Graph Representation Learning Algorithm.

	Parameters:
	-----------

	- **encoder** (Optional[:class:`torch.nn.Module`]): 
  		The encoder to be trained.

	- **hidden_channels** (int):
		Number of hidden channels for the encoder.

	- **readout** (Optional[Union[Callable, torch.nn.Module]]):
		the readout function to obtain the summary emb. of the entire graph. (default: AvgReadout())

	- **loss_function** (Optional[torch.nn.Module]):
		The loss function to be used. (default: LocalGlobalLoss())

	- **gamma** (float):
		The gamma value. (default: 0.1)

	- **num_layers** (int):
		Number of layers. (default: 1)

	- **prior** (bool):
		Whether to use prior. (default: False)



References
^^^^^^^^^^^^^^^
*InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization*. https://arxiv.org/abs/1908.01000


GraphCL
---------------------
Same to the one in General Algorithms section.


GraphMAE
---------------------

Graph Message Aggregation with Context (GraphMAC) is an advanced framework designed to enhance graph representation learning by integrating contextual information into the message-passing process. As graphs become increasingly prevalent across diverse applications—from social networks to biological systems—the need for effective methods that capture both local and global contextual nuances has emerged. GraphMAC addresses this need by rethinking the way information is aggregated within graph structures.
The central premise of GraphMAC lies in its innovative approach to message aggregation, which focuses on incorporating contextual factors to improve the quality of node representations. Traditional graph neural networks often rely on simple aggregation functions that may overlook critical contextual details, leading to suboptimal learning outcomes. GraphMAC aims to bridge this gap by providing a framework that captures rich contextual information during the message-passing phase, resulting in more expressive and meaningful embeddings.

Key Concepts
^^^^^^^^^^^^^^^^^
1. **Contextual Message Aggregation**: GraphMAC introduces a context-aware message aggregation mechanism that enriches the information exchanged between nodes. By considering various contextual signals—such as node attributes, edge types, and neighborhood structures—GraphMAC enhances the representation quality of each node.

2. **Adaptive Weighting**: The framework employs adaptive weighting strategies that dynamically adjust the importance of different messages based on the context. This allows the model to prioritize more relevant information during aggregation, leading to improved node representations.

3. **Scalability and Efficiency**: GraphMAC is designed to be scalable and efficient, making it suitable for large graphs. Its ability to capture complex relationships without incurring significant computational overhead ensures that it can be applied to real-world datasets effectively.

Methodology
^^^^^^^^^^^^^^^^^
The GraphMAC methodology consists of several key steps:

- **Context Extraction**: The first step involves extracting contextual information from the graph, including node features and structural relationships. This context serves as the basis for enhancing message aggregation.

- **Message Passing with Context**: During the message-passing phase, GraphMAC incorporates the extracted context into the aggregation process. This allows for a more nuanced understanding of the relationships between nodes, facilitating richer embeddings.

- **Learning and Optimization**: The model is trained using optimization techniques that focus on maximizing the quality of the learned representations. By integrating context into the learning process, GraphMAC refines node embeddings iteratively.


API Reference in PyG-SSL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: EncoderDecoder(GNN, in_channels=1433, hidden_channels=512, enc_dec="encoding",  num_hidden=256,  num_layers=2, dropout=0.2, activation='prelu', residual=False, norm=None, nhead=4, nhead_out=1, attn_drop=0.1, negative_slope=0.2, concat_out=True)

	An encoder-decoder class for masked autoencoder.

	Parameters:
	-----------

	- **GNN** (Optional[:class:`torch.nn.Module`]):
		The GNN to be used.  

	- **in_channels** (int):
		Number of input features of the input dataset. Default is 1433.

	- **hidden_channels** (int):
		Number of hidden channels for the encoder. Default is 512.

	- **enc_dec** (str):
		The encoder-decoder type. Default is "encoding".

	- **num_hidden** (int):
		Number of hidden units. Default is 256.
	
	- **num_layers** (int):
		Number of layers. Default is 2.
	
	- **dropout** (float):
		Dropout rate. Default is 0.2.
	
	- **activation** (str):
		Activation function. Default is "prelu".

	- **residual** (bool):
		Whether to use residual. Default is False.

	- **norm** (Optional[:class:`torch.nn.Module`]): 
		The normalization layer. Default is None.

	- **nhead** (int):
		Number of heads in multi-head attention. Default is 4.
	
	- **nhead_out** (int):
		Number of heads in multi-head attention for output. Default is 1.

	- **attn_drop** (float):
		Dropout rate for attention. Default is 0.1.

	- **negative_slope** (float):
		Negative slope for LeakyReLU. Default is 0.2.

	- **concat_out** (bool):
		Whether to concatenate the output. Default is True.





.. class:: GraphMAE(encoder: torch.nn.Module, decoder: torch.nn.Module, hidden_channels: int, concat_hidden: bool = False, loss_function: Optional[torch.nn.Module] = None, argument = None)

	Parameters:
	-----------

	- **encoder** (Optional[:class:`torch.nn.Module`]):
		The encoder to be trained.

	- **decoder** (Optional[:class:`torch.nn.Module`]):
		The decoder to be trained.

	- **hidden_channels** (int):
		Number of hidden channels for the encoder.

	- **concat_hidden** (bool):
		Whether to concatenate hidden. Default is False.

	- **loss_function** (Optional[torch.nn.Module]):
		The loss function to be used. Default is None.

	- **argument** (Optional[Dict[str, Any]]):
		The argument to be used. Default is None.


	
References
^^^^^^^^^^^^^^^^^
GraphMAE: Self-Supervised Masked Graph Autoencoders. https://arxiv.org/abs/2205.10803