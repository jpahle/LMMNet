# LmmNet

Here we blend the classical theory of Linear Multi-step Method with machine learning and neural networks (hence LmmNet). The convergence properties of LMM for learning dynamics has been studied by [Keller & Du, 2020](https://arxiv.org/abs/1912.12728).

## Comparison with previous approach

One difference is in the design choice: previous approach trained a different model for every species (dependent variable) considered. That is, the model assumed a mapping from multi-species concentrations X to single species derivatives y. The training is the repeated for all 10 metabolites as in [Costello & Martin, 2018](https://www.nature.com/articles/s41540-018-0054-3).

```
For i in 1 to 10:
    Train model to approximate the function mapping
    Metabolite1, Metabolite2, ..., Protein1, Protein2, ... -> Derivative of Metabolite i
```

However, in LmmNet, we reconstruct the dynamics of all species using a single function mapping

`species1, species2, species3 -> species1, species2, species3`

Advantages of LmmNet:
* multi-step instead of single-step
* avoids artificial computation of the derivatives to create suitable training data

Disadvantages of LmmNet:
* assumes regularly sampled time-series data (obvious solution: preprocessing)

## Milestones

1. Study the stability behaviour of LmmNet for the 2-D oscillator problem, as in [Keller & Du, 2020](https://arxiv.org/abs/1912.12728)
2. Study LmmNet for [2-D Bier model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1300712/): different families of LMM, different number of steps, different time steps/grid sizes, how noise affects the predictions, LmmNet vs analytical solution, oscillatory vs bifurcation problems, different layers/units
3. Extend LmmNet to use other machine learning methods, not just neural networks
4. Inference of mechanistic insights (explainability)
5. Extension to handle missing data and irregular time-series data
6. Deep Learning for solving ODE (symbolic integration)
7. Uncertainty Estimation

## Pending Tasks

* Reproduce Hopf bifurcation and 6-D glycolysis results
* Follow up with Keller et al. -- clarify how to choose $\hat{g}$ and their future direction.
* BioQuant internal seminar (16 April 2020)