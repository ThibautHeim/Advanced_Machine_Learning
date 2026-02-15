# Advanced ML : Programming Exercises - Week 2

## Exercise 2.4

> In the provided code (flow.py), the class MaskedCouplingLayer implements
the masked coupling layer from Real NVP (Dinh et al. 2017), **but it does not implement
the forward transformation, the inverse transformation and the corresponding calculations
of the log determinant of the Jacobian.** 

>In this exercise you should complete the following two functions such that:
>* `MaskedCouplingLayer.forward(...)` returns $T(z)$ and log det $\log|detJ_T(z)|$.>
>* `MaskedCouplingLayer.inverse(...)` returns $T^{−1}(z′)$ and $log |det J_T^{-1} (z′)|$.  

>Use the *TwoGaussians* datasets (c.f., figure 1) for testing the model.   
Adjust the number
of coupling layers and the architecture of the networks to get a good fit to the density
(by qualitative assessment).   
Make sure to write your implementation such that it works
on data of more than two dimensions.  

>**Optional:** Can you also fit a flow to the Chequerboard dataset? It is difficult to find an
architecture that gives a good fit.