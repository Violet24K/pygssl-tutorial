Installation
============

PyG-SSL is a Python package that builds on built upon `PyTorch <https://pytorch.org>`_ and `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_.
It has been tested for Python **3.x** and PyTorch **x.x.x.** on Linux and Windows. It is recommended to use a virtual environment to install the package.

.. code-block:: console

   (.venv) $ pip install pyg-ssl


As PyG-SSL requires the presence of certain prerequisites, installing pyg-ssl in an arbitrary virtual environment may produce some conflicts. We provide a full 
installation script that will install all the necessary dependencies: **ToDo**

.. code-block:: console

   conda create -n pyg-ssl python=3.8
   conda activate pyg-ssl
   conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
   pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
   pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
   pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
   pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
   pip install torch-geometric
   pip install pyg-ssl


The installation can be verified by

>>> import pyg-ssl

