## OverICA

the R package `OverICA` estiamtes overcompelte indepedent components, meanign it estimates more altent variables then there are observed variables in the data. 

$$L = Matrix_{n*p)$$


It does so by levraging the differences betwene the observed generalized covariance matrices and means in the data and the model implied generalized covariance matrices and means. This is based on ideas developed by Podosinnikova et al. (2019). Unlike Podosinnikova et al. I use back propagation to estimate the paprameters.








References:

Podosinnikova, A., Perry, A., Wein, A. S., Bach, F., d’Aspremont, A., & Sontag, D. (2019, April). Overcomplete independent component analysis via SDP. In The 22nd international conference on artificial intelligence and statistics (pp. 2583-2592). PMLR.
