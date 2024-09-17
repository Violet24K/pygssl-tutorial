PyG-SSL Training
=====================

Trainer
-------------------

.. class:: BaseTrainer(method: BaseMethod, data_loader: DataLoader, save_root: str = "./ckpt", device: Union[str, int] = "cpu")

    Base class for self-supervised learning methods.

    Parameters:
    -----------

    - **method** (BaseMethod): 
        the entire method, including encoders and other components (e.g. discriminators)

    - **data_loader** (DataLoader):
        the data loader for the training data

    - **save_root** (str):
        the root directory to save the model checkpoints

    - **device** (str):
        the device to run the training process

    .. method:: train()

        Train the model.

    .. method:: save(path: optinal[str] = None)

        Save the model.

    .. method:: load(path: optinal[str] = None)
            
        Load the model parameters.




.. class:: SimpleTrainer(method: BaseMethod, data_loader: DataLoader, lr: float = 0.001, weight_decay: float = 0.0, n_epochs: int = 10000, patience: int = 50, device: Union[str, int] = "cuda:0", save_root: str = "./ckpt", dataset=None)

    A simple trainer for self-supervised learning methods.

    Parameters:
    -----------

    - **method** (BaseMethod): 
        the entire method, including encoders and other components (e.g. discriminators)

    - **data_loader** (DataLoader):
        the data loader for the training data

    - **lr** (float):
        the learning rate for the optimizer

    - **weight_decay** (float):
        the weight decay for the optimizer

    - **n_epochs** (int):
        the number of epochs for training

    - **patience** (int):
        the patience for early stopping

    - **device** (str):
        the device to run the training process

    - **save_root** (str):
        the root directory to save the model checkpoints

    - **dataset** (Optional[Dataset]):
        the dataset for the training data

    .. method:: train()

        Train the model.

    .. method:: save(path: optinal[str] = None)

        Save the model.

    .. method:: load(path: optinal[str] = None)
            
        Load the model parameters.





.. class:: NonContrastTrainer(method: BaseMethod, data_loader: DataLoader, lr: float = 0.001, weight_decay: float = 0.0, n_epochs: int = 10000, patience: int = 50, device: Union[str, int] = "cuda:0", use_ema: bool = False, moving_average_decay: float = 0.9, save_root: str = "./ckpt", dataset=None)

    A trainer for self-supervised learning methods that are not contrastive (BGRL and AFGRL).

    Parameters:
    -----------

    - **method** (BaseMethod): 
        the entire method, including encoders and other components (e.g. discriminators)

    - **data_loader** (DataLoader):
        the data loader for the training data

    - **lr** (float):
        the learning rate for the optimizer

    - **weight_decay** (float):
        the weight decay for the optimizer

    - **n_epochs** (int):
        the number of epochs for training

    - **patience** (int):
        the patience for early stopping

    - **device** (str):
        the device to run the training process

    - **use_ema** (bool):
        whether to use exponential moving average

    - **moving_average_decay** (float):
        the decay rate for the moving average

    - **save_root** (str):
        the root directory to save the model checkpoints

    - **dataset** (Optional[Dataset]):
        the dataset for the training data

    .. method:: train()

        Train the model.

    .. method:: save(path: optinal[str] = None)

        Save the model.

    .. method:: load(path: optinal[str] = None)
            
        Load the model parameters.




.. class:: GCATrainer(method: BaseMethod, data_loader: DataLoader, lr: float = 0.001, weight_decay: float = 0.0, n_epochs: int = 5000, patience: int = 50, drop_scheme: str = 'degree', dataset_name: str = 'WikiCS', device: Union[str, int] = "cuda:1", save_root: str = "./ckpt")

    A trainer for GCA.

    Parameters:
    -----------

    - **method** (BaseMethod): 
        the entire method, including encoders and other components (e.g. discriminators)

    - **data_loader** (DataLoader):
        the data loader for the training data

    - **lr** (float):
        the learning rate for the optimizer

    - **weight_decay** (float):
        the weight decay for the optimizer

    - **n_epochs** (int):
        the number of epochs for training

    - **patience** (int):
        the patience for early stopping

    - **drop_scheme** (str):
        the drop scheme for the data augmentation

    - **dataset_name** (str):
        the name of the dataset

    - **device** (str):
        the device to run the training process

    - **save_root** (str):
        the root directory to save the model checkpoints

    .. method:: train()

        Train the model.

    .. method:: save(path: optinal[str] = None)

        Save the model.

    .. method:: load(path: optinal[str] = None)
            
        Load the model parameters.







Loss
-------------------

.. class:: LocalGlobalLoss(in_channels, sim_function,)

    Estimate the negative mutual information loss for the specified neural similarity function (sim_function) and the
    inputs x, y, x_ind, y_ind. (x, y) is sampled from the joint distribution (x, y)~p(x, y). x_ind and y_ind are
    independently sampled from their marginal distributions x_ind~p(x), y_ind~p(y).
    Reference: Deep Graph Infomax.

    Parameters:
    -----------

    - **in_channels** (int): 
        input channels to the neural mutual information estimator.

    - **sim_function** (Callable):
        the neural similarity measuring function. The default is the bilinear similarity function used by DGI


    .. method:: forward(l_enc, g_enc, batch, measure)

        Compute the negative mutual information loss.

        Args:
        -------
        - **l_enc** (torch.Tensor): 
            the local encoder output

        - **g_enc** (torch.Tensor):
            the global encoder output

        - **batch** (torch.Tensor):
            the batch of nodes
        
        - **measure** (str):
            the measure to compute the loss. The default is "local_global"



.. class:: SCELoss()

    .. method:: forward(x, y, alpha)

        Compute the supervised contrastive loss.

        Args:
        -------
        - **x** (torch.Tensor): 
            the input tensor

        - **y** (torch.Tensor):
            the target tensor

        - **alpha** (float):
            the temperature parameter


.. class:: SIGLoss()

    .. method:: forward(x, y)

        Compute the supervised infomax loss.

        Args:
        -------
        - **x** (torch.Tensor): 
            the input tensor

        - **y** (torch.Tensor):
            the target tensor



.. class:: NegativeMI(in_channels: Optional[int] = None, sim_func: Optional[torch.nn.Module] = None)

    Negative Mutual Information.
    Estimate the negative mutual information loss for the specified neural similarity function (sim_function) and the
    inputs x, y, x_ind, y_ind. (x, y) is sampled from the joint distribution (x, y)~p(x, y). x_ind and y_ind are
    independently sampled from their marginal distributions x_ind~p(x), y_ind~p(y).
    Reference: Deep Graph Infomax.

    Parameters:
    -----------

    - **in_channels** (int): 
        input channels to the neural mutual information estimator.

    - **sim_function** (Callable):
        the neural similarity measuring function. The default is the bilinear similarity function used by DGI





Evaluation
-------------------

.. class:: K_Means(k: int = 8, average_method: str = "arithmetic", init = "k-means++", n_init = 10, max_iter: int = 300, tol: float = 1e-4, verbose: int = 0, random_state = None, copy_x: bool = True, algorithm: str = "auto", n_run: int = 50, device: Union[str, int] = "cuda")

    K-Means clustering.

    Parameters:
    -----------

    - **k** (int): 
        the number of clusters

    - **average_method** (str):
        the method to compute the average of the clusters

    - **init** (str):
        the initialization method

    - **n_init** (int):
        the number of initializations

    - **max_iter** (int):
        the maximum number of iterations

    - **tol** (float):
        the tolerance

    - **verbose** (int):
        the verbosity

    - **random_state** (int):
        the random state

    - **copy_x** (bool):
        whether to copy the data

    - **algorithm** (str):
        the algorithm to use

    - **n_run** (int):
        the number of runs

    - **device** (str):
        the device to run the training process

    .. method:: __call__(embs, dataset)
            
        Evaluate the embeddings.

        Args:
        -------

        - **embs** (torch.Tensor): 
            the embeddings

        - **dataset** (Dataset):
            the dataset

    .. method:: single_run(embs, labels)

        Evaluate the embeddings (more elegantly).

        Args:
        -------

        - **embs** (torch.Tensor): 
            the embeddings

        - **labels** (torch.Tensor):
            the labels




.. class:: LogisticRegression(lr: float = 0.01, weight_decay: float = 0., max_iter: int = 100, n_run: int = 50, device: Union[str, int] = "cuda")

    Logistic Regression.

    Parameters:
    -----------

    - **lr** (float): 
        the learning rate

    - **weight_decay** (float):
        the weight decay

    - **max_iter** (int):
        the maximum number of iterations

    - **n_run** (int):
        the number of runs

    - **device** (str):
        the device to run the training process

    .. method:: __call__(embs, dataset)
        
        Evaluate the embeddings.

        Args:
        -------

        - **embs** (torch.Tensor): 
            the embeddings

        - **dataset** (Dataset):
            the dataset

    .. method:: single_run(embs, labels, train_mask, val_mask, test_mask)

        Evaluate the embeddings (more elegantly).

        Args:
        -------

        - **embs** (torch.Tensor): 
            the embeddings

        - **labels** (torch.Tensor):
            the labels

        - **train_mask** (torch.Tensor):
            the train mask

        - **val_mask** (torch.Tensor):
            the validation mask

        - **test_mask** (torch.Tensor):
            the test mask



.. class:: RandomForestClassifier(search: bool = False, n_estimators: int = 1, criterion: str = 'gini', max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 1, min_weight_fraction_leaf: float = 0.0, \
        max_features: Union[int, float, str, None] = 'auto', max_leaf_nodes: int = None, min_impurity_decrease: float = 0.0, bootstrap: bool = True, obb_score: bool = False, n_jobs: int = 1, random_state: Union[int, None] = None, \
        verbose: int = 0, warm_start: bool = False, class_weight: dict = None, n_run: int = 50, device: Union[str, int] = "cuda")

    The Random Forest Classifier.

    Parameters: 
    -----------

    - **search** (bool): 
        whether to search for the best hyperparameters

    - **n_estimators** (int):
        the number of trees in the forest

    - **criterion** (str):
        the function to measure the quality of a split

    - **max_depth** (int):
        the maximum depth of the tree

    - **min_samples_split** (int):
        the minimum number of samples required to split an internal node

    - **min_samples_leaf** (int):
        the minimum number of samples required to be at a leaf node

    - **min_weight_fraction_leaf** (float):
        the minimum weighted fraction of the sum total of weights

    - **max_features** (Union[int, float, str, None]):
        the number of features to consider when looking for the best split

    - **max_leaf_nodes** (int):
        the maximum number of leaf nodes

    - **min_impurity_decrease** (float):
        the minimum impurity decrease required for a split

    - **bootstrap** (bool):
        whether bootstrap samples are used when building trees

    - **obb_score** (bool):
        whether to use out-of-bag samples to estimate the generalization accuracy

    - **n_jobs** (int):
        the number of jobs to run in parallel

    - **random_state** (int):
        the random state

    - **verbose** (int):
        the verbosity

    - **warm_start** (bool):
        whether to reuse the solution of the previous call to fit

    - **class_weight** (dict):
        the class weight

    - **n_run** (int):
        the number of runs

    - **device** (str):
        the device to run the training process

    .. method:: __call__(embs, dataset)
        
        Evaluate the embeddings.

        Args:
        -------

        - **embs** (torch.Tensor): 
            the embeddings

        - **dataset** (Dataset):
            the dataset

    .. method:: single_run(embs, labels, train_mask, val_mask, test_mask)

        Evaluate the embeddings (more elegantly).

        Args:
        -------

        - **embs** (torch.Tensor): 
            the embeddings

        - **labels** (torch.Tensor):
            the labels

        - **train_mask** (torch.Tensor):
            the train





.. class:: SVCRegression(C: float = 1.0, search: bool = True, kernel: str = 'rbf', degree: int = 3,gamma: str='auto', coef0: float = 0.0, shrinking: bool = True, probability: bool = False, tol: float = 0.001, cache_size: int = 200, \
            class_weight: dict = None, verbose: bool = False, max_iter: int = -1, decision_function_shape: str = 'ovr', random_state: int = None, n_run: int = 50, device: Union[str, int] = "cuda")

    The Support Vector Classifier.

    Parameters:
    -----------

    - **C** (float): 
        the regularization parameter

    - **search** (bool):
        whether to search for the best hyperparameters

    - **kernel** (str):
        the kernel type

    - **degree** (int):
        the degree of the polynomial kernel function

    - **gamma** (str):
        the kernel coefficient

    - **coef0** (float):
        the independent term in the kernel function

    - **shrinking** (bool):
        whether to use the shrinking heuristic

    - **probability** (bool):
        whether to enable probability estimates

    - **tol** (float):
        the tolerance

    - **cache_size** (int):
        the cache size

    - **class_weight** (dict):
        the class weight


    .. method:: __call__(embs, dataset)
        
        Evaluate the embeddings.

        Args:
        -------

        - **embs** (torch.Tensor): 
            the embeddings

        - **dataset** (Dataset):
            the dataset

    .. method:: single_run(embs, labels, train_mask, val_mask, test_mask)

        Evaluate the embeddings (more elegantly).

        Args:
        -------

        - **embs** (torch.Tensor): 
            the embeddings

        - **labels** (torch.Tensor):
            the labels

        - **train_mask** (torch.Tensor):
            the train


.. class:: SimSearch(sim_list: list = [5, 10, 20, 50, 100], n_run: int = 50, device: Union[str, int] = "cuda")

    Similarity Search.

    Parameters:
    -----------

    - **sim_list** (list): 
        the list of similarity values

    - **n_run** (int):
        the number of runs

    - **device** (str):
        the device to run the training process

    .. method:: __call__(embs, dataset)
        
        Evaluate the embeddings.

        Args:
        -------

        - **embs** (torch.Tensor): 
            the embeddings

        - **dataset** (Dataset):
            the dataset

    .. method:: single_run(embs, labels)

        Evaluate the embeddings (more elegantly).

        Args:
        -------

        - **embs** (torch.Tensor): 
            the embeddings

        - **labels** (torch.Tensor):
            the labels