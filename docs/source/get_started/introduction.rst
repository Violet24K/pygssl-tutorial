Introduction
============


SSL and PyG-SSL
-----------------
(Semi-)Sueprvised Learning algorithms often rely on heavy label information, which is expensive to obtain. 
Self-Supervised Learning (SSL) is a promising alternative, which aims to learn representations from the data itself. 
SSL has been widely studied in the context of images and text, but has only recently been extended to graph-structured data. 
PyG-SSL is a PyTorch Geometric-based library for self-supervised learning on graphs. 
It provides a collection of popular benchmark datasets, evaluation metrics, and self-supervised learning algorithms, which can be used to learn node and graph representations 
from graph-structured data. PyG-SSL is built on top of PyTorch Geometric, which is a popular library for graph neural networks. 
PyG-SSL is designed to be easy to use, and can be easily integrated into existing PyTorch Geometric-based projects.


Citing
---------------------

If you find PyG-SSL useful in your research, please consider citing the following paper:

.. code-block:: bibtex

   @article{pygssl,
     title={PyG-SSL: A PyTorch Geometric-based Library for Self-Supervised Learning on Graphs},
     author={Anonymous Authors},
     journal={arXiv preprint arXiv: xxxx.xxxxx},
     year={2024}
   }



PyG-SSL not only includes the algorithms themselves, but also the benchmark datasets and evaluation metrics. The integrated datasets are:

**ToDo**
