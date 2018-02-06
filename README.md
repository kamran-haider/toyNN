[![Build Status](https://travis-ci.org/kamran-haider/toyNN.svg?branch=master)](https://travis-ci.org/kamran-haider/toyNN)

======
toyNN
======


.. image:: https://img.shields.io/pypi/v/pymcce.svg
        :target: https://pypi.python.org/pypi/pymcce

.. image:: https://img.shields.io/travis/kamran-haider/pymcce.svg
        :target: https://travis-ci.org/kamran-haider/pymcce

.. image:: https://readthedocs.org/projects/pymcce/badge/?version=latest
        :target: https://pymcce.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/kamran-haider/pymcce/shield.svg
     :target: https://pyup.io/repos/github/kamran-haider/pymcce/
     :alt: Updates


A bare-bones implementation of deep learning neural networks. `toyNN` is not comparable to powerful deep learning libraries
such as as `Tensorflow`, `Keras`, `PyTorch` and many others. It's goal is not to train production quality deep learning 
neural networks. It is a personal project born out of a desire to:

* Understanding deep learning neural networks by coding them up and gain insights into various tricks that make them so powerful.
* Gain practice in prototyping and shipping machine learning algorithms utilzing python/git/CI ecosystem.
* Develop an understanding of productive API design, by achieving tight integration of `toyNN` with `scikit-learn`'s estimator API.



Features
--------
* Sigmoid and ReLU layers
* Batch Gradient Descent
* Constant, He and Xavier weight initializations


TODO
----

* Dropout
* L1 and L2 Regularization
* Mini-batch gradient descent



License
-------

* MIT license

Documentation
-------------
* https://toyNN.readthedocs.io.



Credits
---------

The implementation of deep learning neural networks is this package is inspired from lectures and codes in Andrew Ng's
Deep Learning Specialization on Coursera. 
