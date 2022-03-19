---
title: PredFlow
tagline: Predictive coding using tensorflow
---

Maximum likelihood estimation on factorized gaussian models can be realized through local prediction error minimization. This observation lead to the development of _predictive coding_ theories of cortical computation and cognition. 
Very good introduction to these theories 
can be found from very different points of view including [theoretical neuroscience](https://www.nature.com/articles/nrn2787),
[experimental neuroscience](https://www.sciencedirect.com/science/article/pii/S0896627318308572),
[philosophy of mind](https://predictive-mind.net/papers/vanilla-pp-for-philosophers-a-primer-on-predictive-processing),
[neuropsychology](https://www.sciencedirect.com/science/article/pii/S0006322318315324),
[cognitive science](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/whatever-next-predictive-brains-situated-agents-and-the-future-of-cognitive-science/33542C736E17E3D1D44E8D03BE5F4CD9#article) and
[machine learning](https://arxiv.org/abs/2107.12979) (*citation does not mean endorsement*). 

This repository of tutorials and short software pieces is mainly a support for my PhD and is not intended as a general predictive coding simulation platform.
The topics that I will (hopefully) cover during my PhD includes 
 - **The geometry of precision in predictive coding**
 - **Lateral dynamics in predictive coding - correlations and sparsity**
 - **Semantization through multimodal predictions**
 - **The place of predictive coding in the brain's algorithm**

These will then be the main focus here, at least when it comes to my contribution.

If you don't know where to start, you can look at simple examples of [classification](simpleclassification.md) and [generation](simplegeneration.md) with predictive coding on a multi-layer perceptron architecture. 

As a very simple benchmark, training a 784-256-64-10 multilayer perceptron on 60000 MNIST digits with 10 steps of inference before each weight update for 10
epochs with minibatches of size 100 takes approximately 15-20 seconds on mid-range consumer hardware (CPU: AMD Ryzen 7 3700X (16) @ 3.600GHz, GPU: NVIDIA GeForce RTX 2070 SUPER).

## Content
 - [A simple example of using predictive coding to generate MNIST digits](simplegeneration.md)
 - [A simple example of using predictive coding to classify MNIST digits](simpleclassification.md)
 - [Visualization of the computation graph using tensorboard](tensorboardexample.md)