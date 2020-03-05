# LMMNet

Building on our previous work on inference/discovery/prediction of dynamics from time-series data, we now seek to avoid having to artifically compute the derivatives from training data via the use of Multistep methods. Intuitively, this can produce more robust results because the ML model can learn the local dynamics from several time steps, rather than predicting the derivative at any time point based only on the value at that time point.

Moreover, the convergence properties for the disovery of dynamics using LMM has also been studied [Keller & Du, 2020](https://arxiv.org/abs/1912.12728).

## Milestones

1. Show that lmmNet is able to recover the dynamics of the cubic oscillator (textbook problem in systems identification).
2. Reproduce results from [Keller & Du, 2020](https://arxiv.org/abs/1912.12728) for the three major schemes of LMM (AM, AB, BDF) using different number of steps.
3. Apply lmmNet to complex non-linear dynamics in biochemical systems
4. Design a way to characterize errors? CV?
5. Show that it works with different ML models
6. Noisy data?
7. Explainable AI?
8. Deep Learning for solving ODE (symbolic integration)

Practical issues:
* noise
* complete feature space unknown
* missing time points, not enough time points