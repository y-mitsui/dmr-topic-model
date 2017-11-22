#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free
import numpy as np

cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil


cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive")
    return lda_lgamma(x)


cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin


def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, double[:, :] nzw, double[:, :] ndz, double[:] nz,
                   double[:] eta, double[:] rands, double[:, :] sample_X, double[:, :] lamb, double[:] weights):
    cdef int i, k, w, d, z, z_new
    cdef double r, dist_cum
    cdef int N = WS.shape[0]
    cdef int n_rand = rands.shape[0]
    cdef int n_topics = nz.shape[0]
    cdef double eta_sum = 0
    cdef double* dist_sum = <double*> malloc(n_topics * sizeof(double))
    if dist_sum is NULL:
        raise MemoryError("Could not allocate memory during sampling.")

    for i in range(eta.shape[0]):
        eta_sum += eta[i]

    for i in range(N):
        w = WS[i]
        d = DS[i]
        z = ZS[i]

        nzw[z, w] -= weights[d]
        ndz[d, z] -= weights[d]
        nz[z] -= weights[d]
        #dec(nzw[z, w])
        #dec(ndz[d, z])
        #dec(nz[z])

        dist_cum = 0
        for k in range(n_topics):
            # eta is a double so cdivision yields a double
            dist_cum += (nzw[k, w] + eta[w]) / (nz[k] + eta_sum) * (ndz[d, k] + np.exp(np.dot(lamb[k], sample_X[d])))
            dist_sum[k] = dist_cum

        r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
        z_new = searchsorted(dist_sum, n_topics, r)

        ZS[i] = z_new
        #inc(nzw[z_new, w])
        #inc(ndz[d, z_new])
        #inc(nz[z_new])
        nzw[z_new, w] += weights[d]
        ndz[d, z_new] += weights[d]
        nz[z_new] += weights[d]

    free(dist_sum)


cpdef double _loglikelihood(double[:, :] nzw, double[:, :] ndz, double[:] nz, double[:] nd, double eta, double[:, :] sample_X, double[:, :] lamb):
    cdef int k, d
    cdef int D = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int vocab_size = nzw.shape[1]

    cdef double ll = 0

    # calculate log p(w|z)
    cdef double lgamma_eta, lgamma_alpha
    
    lgamma_eta = lgamma(eta)

    ll += n_topics * lgamma(eta * vocab_size)
    for k in range(n_topics):
        ll -= lgamma(eta * vocab_size + nz[k])
        for w in range(vocab_size):
            # if nzw[k, w] == 0 addition and subtraction cancel out
            if nzw[k, w] > 0:
                ll += lgamma(eta + nzw[k, w]) - lgamma_eta

    # calculate log p(z)
    for d in range(D):
        for k in range(n_topics):
            alpha_val = np.exp(np.dot(sample_X[d], lamb[k]))
            ll += (lgamma(alpha_val) -
                lgamma(alpha_val + nd[d]))
            if ndz[d, k] > 0:
                ll += lgamma(alpha_val + ndz[d, k]) - lgamma(alpha_val)
    return ll
