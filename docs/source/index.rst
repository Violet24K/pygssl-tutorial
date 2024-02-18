GraphSSL documentation
===================================

**GraphSSL** is a Python library built upon `PyTorch <https://pytorch.org>`_.
It integrates state-of-the-art algorithms for Self-Supervised Learning on Graphs and offers simple APIs to 
call the algorithms from a variety of high-quality research papers. GraphSSL can be helpful for many downstream tasks, depending on the dataset
properties, as it contains many self-supervised algorithms for different kinds of graphs:

   **General/Regular Graphs**

   **Heterogeneous/Multiplex/Multiview Graphs**

   **Molecular Graphs**

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

   get_started/tutorial
   get_started/toadd


.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   tutorials/general.rst
   tutorials/non-contrastive.rst
   tutorials/heterogeneous.rst



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 📚 Reference

   reference/trainer.rst
   reference/loader.rst