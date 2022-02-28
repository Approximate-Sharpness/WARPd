# WARPd
Code for the Weighted, Accelerated and Restarted Primal-dual algorithm. This algorithm achieves stable linear convergence for reconstruction from undersampled noisy measurements under an approximate sharpness condition. See the paper for details.

The paper: "WARPd: A linearly convergent first-order primal-dual algorithm for inverse problems with approximate sharpness conditions"

Contents of code:  

  Main routines:  
        WARPd.m: main routine for the algorithm.  
        WARPdSR.m: noise-blind recovery version based on additional square-root LASSO term  
        WARPd_mc.m: version for matrix completion that uses PROPACK  
        WARPd_reweight.m and WARPdSR_reweight.m: iterative reweighting versions used for final numerical experiments 
        
  Example code of how to use main routines:  
        matrix_completion_example.m  
        shearlet_TVG_example.m  

NB: Currently, the code is setup for the Euclidean case of Example 1 from the paper.