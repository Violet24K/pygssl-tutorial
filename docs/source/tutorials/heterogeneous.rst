Heterogeneous Algorithms
================================


HeCo
---------------------

HeCo (Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning) is an innovative framework designed to effectively learn representations from heterogeneous graphs by leveraging self-supervised learning techniques. In heterogeneous graphs, nodes and edges can belong to multiple types, making it crucial to capture the intricate relationships and interactions between diverse entities. HeCo addresses this challenge by introducing a co-contrastive learning approach that enhances the learning process, allowing for the extraction of rich, informative embeddings.

Introduction
^^^^^^^^^^^^^^^^^
The core idea of HeCo is to utilize co-contrastive learning to enable a graph neural network to learn robust representations from both the structural and attribute information present in heterogeneous graphs. Traditional methods often struggle to effectively represent the complexity of such graphs due to the diverse nature of their components. HeCo overcomes this limitation by applying a dual contrastive loss mechanism that simultaneously contrasts node representations across different views and types, thereby promoting meaningful embeddings that reflect the underlying graph structure.

Key Concepts
^^^^^^^^^^^^^^^^^
1. **Heterogeneous Graph Representation**: HeCo is specifically designed for heterogeneous graphs, recognizing the diversity in node and edge types. This allows the framework to capture the unique relationships and features associated with different entities within the graph.

2. **Co-contrastive Learning**: The framework introduces a co-contrastive learning approach, where representations are learned by contrasting positive pairs (similar entities) and negative pairs (dissimilar entities) across multiple views. This dual mechanism enhances the robustness of the learned embeddings.

3. **Self-supervised Framework**: Operating within a self-supervised learning paradigm, HeCo eliminates the need for labeled data, making it applicable in scenarios where acquiring labels is difficult or costly. This enhances its versatility across various applications.

Methodology
^^^^^^^^^^^^^^^^^
HeCo's methodology involves several key steps:

- **Graph Representation Learning**: The process begins with the generation of diverse views of the heterogeneous graph, allowing for the exploration of different aspects of the data.

- **Contrastive Loss Computation**: HeCo employs a co-contrastive loss function to guide the learning process, maximizing the similarity between positive pairs while minimizing it for negative pairs.

- **Optimization and Refinement**: The model undergoes iterative training to optimize the learned representations, ensuring that the embeddings effectively capture the complexities of the heterogeneous graph structure.

By combining self-supervised learning with a co-contrastive approach, HeCo provides a powerful tool for extracting meaningful representations from heterogeneous graphs, paving the way for advancements in various applications such as recommendation systems, social network analysis, and knowledge graph construction.


API Reference in PyG-SSL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. class:: Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)

	The encoder based on the network schema view. Network schema view is a view that encodes the network structure information. The encoder is based on the GAT model.

	Parameters:
	-----------

	- **hidden_dim** (int): 
  		The hidden dimension of the GAT model.

	- **sample_rate** (float):
		The sample rate of the encoder.
	
	- **nei_num** (int):
		The number of neighbors to sample.
	
	- **attn_drop** (float):
		The dropout rate of the attention mechanism.



.. class:: Mp_encoder(P, hidden_dim, attn_drop)

	The encoder based on meta-path. Meta-path is a sequence of node types that defines the structural information of the heterogeneous graph. The encoder is based on the GAT model.

	Parameters:
	-----------
	- **P** (List[List[int]]): 
  		The meta-path matrix.
	
	- **hidden_dim** (int):
		The hidden dimension of the GAT model.

	- **attn_drop** (float):
		The dropout rate of the attention mechanism.


.. class:: Contrast(hidden_dim, tau, lam)

	The contrastive loss function.

	Parameters:
	-----------
	- **hidden_dim** (int): 
  		The hidden dimension of the GAT model.

	- **tau** (float):
		The temperature parameter of the contrastive loss.
	
	- **lam** (float):
		The regularization parameter of the contrastive loss.



.. class:: HeCo(encoder1: torch.nn.Module, encoder2: torch.nn.Module, feats_dim_list, readout: Union[Callable, torch.nn.Module] = AvgReadout(), loss_function: Optional[torch.nn.Module] = None, data_argument: None = None, hidden_channels: int = 64, feat_drop: float = 0.3, tau: float = 0.9, lam: float = 0.5,)

	The HeCo algorithm

	Parameters:
	-----------
	- **encoder1** (torch.nn.Module): 
  		The encoder based on the network schema view.

	- **encoder2** (torch.nn.Module):
		The encoder based on the meta-path view.

	- **feats_dim_list** (List[int]):
		The dimension of the features of each node type.
	
	- **readout** (Union[Callable, torch.nn.Module]):
		The readout function to generate global embeddings. (default: AvgReadout())

	- **loss_function** (Optional[torch.nn.Module]):
		The loss function to optimize the embeddings. (default: None)

	- **data_argument** (None):
		The data argumentation function. (default: None)

	- **hidden_channels** (int):
		The hidden dimension of the GAT model. (default: 64)

	- **feat_drop** (float):
		The dropout rate of the features. (default: 0.3)
	
	- **tau** (float):
		The temperature parameter of the contrastive loss. (default: 0.9)

	- **lam** (float):
		The regularization parameter of the contrastive loss. (default: 0.5)






References
^^^^^^^^^^^^^^^^^

- Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning. Xiao Wang et al. https://arxiv.org/abs/2105.09111