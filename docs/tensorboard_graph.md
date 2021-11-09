# Visualize the computational graph

One of the advantage of tensorflow is that you can easily visualize the
computational graph that is actually running on your computer
using [tensorboard](https://www.tensorflow.org/tensorboard). 

Here is what the computational graph of `supervised_fullyconnected_stricthierarchy_explicit_pc.learn` for an MLP with 2 hidden layers looks like from a high level:

<a href="https://ibb.co/DLHZ6DX"><img src="https://i.ibb.co/FnLFfbv/full.png" alt="full" border="0"></a>

Of course we can get to a finer level of detail, and inspect for example the initialization phase:
<a href="https://ibb.co/sbwSsMY"><img src="https://i.ibb.co/ckF527n/init.png" alt="init" border="0"></a>

We can see that the initialization consist of a forward sweep through the model, stopping before the last layer since its activity is clamped to targets in this function, and an initalization of prediction errors to zero.

Let us also further inspect the graph of the inference loop:
<a href="https://ibb.co/GTxQrSM"><img src="https://i.ibb.co/8cKgQZx/inferenceloop.png" alt="inferenceloop" border="0"></a>

We can see that, for this function, it is alternating sequentially between prediction error computation and representation update (for 5 timesteps here). We can further inspect prediction errors computation:
<a href="https://ibb.co/McF6MSs"><img src="https://i.ibb.co/jynRgfT/pe.png" alt="pe" border="0"></a>

and remark that it indeed computes `e[i] = r[i+1] - W[i] * f(r[i])` as expected for this model. 

A further inspection of representation update illustrates the computation `r[i] += ir * (-e[i-1] + tranpose(W[i]) * e[i] .* f'(r[i]))`:
<a href="https://ibb.co/mzhPh58"><img src="https://i.ibb.co/YjfVfBZ/reprupdate.png" alt="reprupdate" border="0"></a>

Finally we can inspect the weight update computational graph, illustrating the computation `W[i] += lr * (e[i] * transpose(f(r[i])))`
<a href="https://ibb.co/X4ZHvWt"><img src="https://i.ibb.co/PYZkfrg/weightupdate.png" alt="weightupdate" border="0"></a>

A particularly important feature of the three last core computational graphs is that operations for each layers and weight matrices are executed in parallel (nodes on the same horizontal level in tensorboard graphs are executed in parallel). This certainly illustrates an important property of predictive coding, namely that it is highly parallelizable across layers, since there is no need to backpropagate gradients (because representations and parameters update are based on _local_ prediction errors).