PyRMLE

**Authors**: Mendoza, E., Dunker, F., Reale, M.

PyRMLE is a Python Module that implements Regularized Maximum Likelihood Estimation for the Random Coefficients Model.

The package's implementation of regularized maximum likelihood is limited to applications with up to two regressors for the random coefficients model with intercept, and up to three regressors for a model without intercept. 

There are two main functions used to implement regularized maximum likelihood estimation using PyRMLE, namely: (1) transmatrix() which executes a finite-volume algorithm that is akin to an algebraic reconstruction of the Radon Transform, and (2) rmle() which is a wrapper function of the scipy.optimize.minimize() function that solves the constrained regularized maximum likelihood problem.

For more details check the github repository here: https://github.com/eae-mendoza/PyRMLE
