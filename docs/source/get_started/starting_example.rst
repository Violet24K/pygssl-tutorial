Starting Example
============

We introduce the fundamental concepts of PyG-SSL through self-contained examples. In this example, we will show how to obtain
and evaluate the embeddings of TUDataset using the InfoGraph self-supervised learning.

Load the Data and Default Configuration
---------------------------------------------------
Our Algorithms APIs are compatible with torch_geometric.datasets. We can easily load the datasets by specifying the name of the dataset. Also, we provide
default configurations, including hyperparameters for algorithms. 

.. code-block:: python

    import os
    from src.config import load_yaml
    from torch_geometric.datasets import TUDataset

    config = load_yaml('./configuration/infograph_mutag.yml')
    current_folder = os.path.abspath('')
    path = os.path.join(current_folder, config.dataset.root, config.dataset.name)

    dataset = TUDataset(path, name=config.dataset.name)
    data_loader = DataLoader(dataset, batch_size=config.dataset.batch_size)


Create and Train the Model
----------------------------------
We can create the model by specifying the name of the algorithm using the pre-defined configuration files. The configurations can be manually modified.
We provide simple APIs to call a trainer to train the model.

.. code-block:: python

    import torch
    from torch_geometric.nn import GINConv
    from src.methods.infograph import InfoGraph, Encoder
    from src.trainer import SimpleTrainer

    device = torch.device("cuda:{}".format(config.gpu_idx) if torch.cuda.is_available() and config.use_cuda else "cpu")
    encoder = Encoder(in_channels=in_channels, hidden_channels=config.model.hidden_channels,
                    num_layers=config.model.n_layers, GNN=GINConv)
    method = InfoGraph(encoder=encoder, hidden_channels=config.model.hidden_channels, num_layers=config.model.n_layers,
                    prior=False)

    trainer = SimpleTrainer(method=method, data_loader=data_loader, device=device, n_epochs=config.optim.max_epoch)
    trainer.train()


Evaluate the Model
----------------------------------
Our methods provide API :class`get_embs` to obtain the embeddings of the dataset. We can then use the embeddings to evaluate the performance of the model.

.. code-block:: python

    from src.evaluation import LogisticRegression

    method.eval()
    data_pyg = dataset.data.to(method.device)
    y, embs = method.get_embs(data_loader)

    data_pyg.x = embs
    lg = LogisticRegression(lr=config.classifier.base_lr, weight_decay=config.classifier.weight_decay,
                            max_iter=config.classifier.max_epoch, n_run=1, device=device)
    lg(embs=embs, dataset=data_pyg)
    >>>** Val: 87.8070 (5.7418) | Test: 87.8070 (5.7418) **


This is all it takes to implement the InfoGraph self-supervised learning on TUDataset. For each algorithm, we provide complete example that can be found in the "Tutorial" Section.