.. PredFlow documentation master file, created by
   sphinx-quickstart on Mon Nov  1 19:02:35 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PredFlow's documentation!
====================================


Predictive coding has been described as
"the most complete framework to date for explaining perception, cognition, and action in terms of fundamental theoretical 
principles and neurocognitive architecture" (Anil Seth in *The cybernetic Bayesian brain*, 2004) and as "a computational framework for understanding 
cortical function in the context of emerging evidence" (Georg Keller and Thomas Mrsic-Flogel in *Predictive processing: a canonical cortical computation*, 2018).
It is both a computational and process theory that aims at approximating bayesian inference and updating in hierarchical factorized gaussian
generative models using prediction error minimization, in a computationally tractable way that could be 
implemented by neural circuits. This framework invites a reinterpretation of cortical computation from feature detection to
inference on and learning of a generative model. This documentation is of course not the place to unwrap these sentences 
and introduce predictive coding in more details. Very good introduction to this theory 
can be found from very different points of view including `theoretical neuroscience <https://www.nature.com/articles/nrn2787>`_,
`experimental neuroscience <https://www.sciencedirect.com/science/article/pii/S0896627318308572>`_,
`philosophy of mind <https://predictive-mind.net/papers/vanilla-pp-for-philosophers-a-primer-on-predictive-processing>`_,
`neuropsychology <https://www.sciencedirect.com/science/article/pii/S0006322318315324>`_,
`cognitive science <https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/whatever-next-predictive-brains-situated-agents-and-the-future-of-cognitive-science/33542C736E17E3D1D44E8D03BE5F4CD9#article>`_ and
`machine learning <https://arxiv.org/abs/2107.12979>`_. 

This package for now is mainly a support for my PhD and is not intended as a general predictive coding simulation platform, though the
code can certainly be reused as is or with minor tweaks for applications and experiments similar to the one I conduct.
The topics that I will (hopefully) cover during my PhD includes **The geometry of precision in predictive coding**, **Lateral dynamics in predictive coding - correlations and sparsity**, **Semantization through multimodal predictions** and **The place of predictive coding in the brain's algorithm**. These will then be the main focus of this package, at least when it comes to my contribution.

If you want to use this package and don't know where to start, you can look at two simple usage examples, one for `classification on MNIST <https://arnogranier.github.io/PredFlow/_build/html/mnist_experiments.html>`_ and the other for `generating MNIST digits <https://arnogranier.github.io/PredFlow/_build/html/mnist_generation.html>`_.

Current features include vanilla predictive coding and predictive coding with precision weighting of prediction errors on the MNIST dataset. 

As a very simple benchmark, training a 784-256-64-10 multilayer perceptron on 60000 MNIST digits with 10 steps of inference before each weight update for 10
epochs with minibatches of size 100 takes approximately 15-20 seconds on mid-range consumer hardware (CPU: AMD Ryzen 7 3700X (16) @ 3.600GHz, GPU: NVIDIA GeForce RTX 2070 SUPER).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   usage
   

