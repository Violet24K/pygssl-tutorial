PyG-SSL documentation
===================================

**PyG-SSL** is a Python library built upon `PyTorch <https://pytorch.org>`_ and `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_.
It integrates state-of-the-art algorithms for Self-Supervised Learning on Graphs and offers simple APIs to 
call the algorithms from a variety of high-quality research papers. PyG-SSL can be helpful for many downstream tasks, depending on the dataset
properties, as it contains many self-supervised algorithms:

   **Contrastive Algorithms for General Graphs**: `DGI <https://arxiv.org/pdf/1809.10341.pdf>`_, `GCC <https://arxiv.org/pdf/2006.09963.pdf>`_, `GraphCL <https://proceedings.nips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf>`_, 
   `MVGRL <https://proceedings.mlr.press/v119/hassani20a/hassani20a.pdf>`_, `GCA <https://arxiv.org/abs/2010.14945>`_, `JOAO <https://proceedings.mlr.press/v139/you21a.html>`_, 
   `SUGRL <https://ojs.aaai.org/index.php/AAAI/article/view/20748>`_, `MERIT <https://www.ijcai.org/proceedings/2021/0204.pdf>`_, 
   `BGRL <https://arxiv.org/abs/2102.06514>`_, `AFGRL <https://arxiv.org/abs/2112.02472>`_

   **Algorithms for Heterogeneous/Multiplex/Multiview Graphs**: `DMGI <https://arxiv.org/pdf/1911.06750.pdf>`_, `HeCo <https://arxiv.org/pdf/2105.09111.pdf>`_, `MCGC <https://proceedings.neurips.cc/paper/2021/file/10c66082c124f8afe3df4886f5e516e0-Paper.pdf>`_

   **Algorithms for Molecular Graphs**: `InfoGraph <https://openreview.net/pdf?id=r1lfF2NYvH>`_, `GraphCL <https://proceedings.nips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf>`_,
   `AD-GCL <https://openreview.net/forum?id=ioyq7NsR1KJ>`_, `JOAO <https://proceedings.mlr.press/v139/you21a.html>`_, `GraphMAE <https://arxiv.org/pdf/2205.10803.pdf>`_

.. Check out the section for further information, including
.. how to :ref:`installation` the project.

.. note::

   This project is under active development.


.. toctree::
   :maxdepth: 1
   :caption: 🚀 install

   install/installation

.. toctree::
   :maxdepth: 1
   :caption: 💡 Get Started

   get_started/introduction
   get_started/starting_example


.. toctree::
   :maxdepth: 2
   :caption: 💯 Tutorial

   tutorials/general.rst
   tutorials/non-contrastive.rst
   tutorials/heterogeneous.rst
   tutorials/molecure.rst



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 📚 Reference

   reference/dataset.rst
   reference/trainer.rst
   reference/algorithms.rst
  