# coding=utf-8
"""Latent Dirichlet allocation using collapsed Gibbs sampling"""

from __future__ import absolute_import, division, unicode_literals  # noqa
import logging
import sys

from utils import check_random_state, matrix_to_lists
import numpy as np
from scipy.special import digamma
import _lda
from scipy.optimize import fmin_cg


logger = logging.getLogger('lda')

PY2 = sys.version_info[0] == 2
if PY2:
    range = xrange

class LambdaTurning:
    def __init__(self, n_topics, sample_X, ndz_, sigma2):
        self.sample_X = sample_X
        self.ndz_ = ndz_
        self.sigma2 = sigma2
        self.n_topics = n_topics
    
    def lhood(self, lamb):
        pass
    
    def gradLambda(self, lamb):
        lamb = lamb.reshape(self.n_topics, -1)
        v_dot_all = np.dot(lamb, self.sample_X.T)
        dig1_all = []
        for d in range(self.sample_X.shape[0]):
            v_dot_exp = np.exp(v_dot_all[:, d])
            val_sum = np.sum(v_dot_exp)
            val_sum2 = np.sum(self.ndz_[d] + v_dot_exp)
            dig1_all.append(digamma(val_sum) - digamma(val_sum2))
            
        grad = []
        for k in range(self.n_topics):
            tmp_all = []
            for d in range(self.sample_X.shape[0]):
                v_dot = v_dot_all[:, d]
                s_dot = v_dot[k]
                s_dot_exp = np.exp(s_dot)
                val2 = self.ndz_[d, k] + np.exp(s_dot)
                dig = dig1_all[d] + digamma(val2) - digamma(s_dot_exp)
                tmp_all.append(s_dot_exp * dig)
                
            for c in range(self.sample_X.shape[1]):
                grad_val = 0.
                for d in range(self.sample_X.shape[0]):
                    grad_val += self.sample_X[d, c] * tmp_all[d]
                    
                    """
                    if grad_val != grad_val:
                        print("lamb", lamb)
                        print("sample_X[d]", self.sample_X[d])
                        print("val2", val2)
                        print("s_dot_exp", s_dot_exp)
                        print("v_dot_exp", v_dot_exp)
                        print("val_sum", val_sum)
                        print("val_sum2", val_sum2)
                        print("self.ndz_[d, k]", self.ndz_[d, k])
                        print(dig)
                        raise Exception("error grad_val")
                    """
                    
                grad.append((grad_val - lamb[k, c] / self.sigma2) / self.sample_X.shape[0])
            
        if np.any(np.isnan(grad)):
            raise Exception("error", grad)
        
        #print("grad", grad)
        return np.array(grad)

class LDA:
    """Latent Dirichlet allocation using collapsed Gibbs sampling

    Parameters
    ----------
    n_topics : int
        Number of topics

    n_iter : int, default 2000
        Number of sampling iterations

    alpha : float, default 0.1
        Dirichlet parameter for distribution over topics

    eta : float, default 0.01
        Dirichlet parameter for distribution over words

    random_state : int or RandomState, optional
        The generator used for the initial topics.

    Attributes
    ----------
    `components_` : array, shape = [n_topics, n_features]
        Point estimate of the topic-word distributions (Phi in literature)
    `topic_word_` :
        Alias for `components_`
    `nzw_` : array, shape = [n_topics, n_features]
        Matrix of counts recording topic-word assignments in final iteration.
    `ndz_` : array, shape = [n_samples, n_topics]
        Matrix of counts recording document-topic assignments in final iteration.
    `doc_topic_` : array, shape = [n_samples, n_features]
        Point estimate of the document-topic distributions (Theta in literature)
    `nz_` : array, shape = [n_topics]
        Array of topic assignment counts in final iteration.

    Examples
    --------
    >>> import numpy
    >>> X = numpy.array([[1,1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
    >>> import lda
    >>> model = lda.LDA(n_topics=2, random_state=0, n_iter=100)
    >>> model.fit(X) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LDA(alpha=...
    >>> model.components_
    array([[ 0.85714286,  0.14285714],
           [ 0.45      ,  0.55      ]])
    >>> model.loglikelihood() #doctest: +ELLIPSIS
    -40.395...

    References
    ----------
    Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent Dirichlet
    Allocation." Journal of Machine Learning Research 3 (2003): 993–1022.

    Griffiths, Thomas L., and Mark Steyvers. "Finding Scientific Topics."
    Proceedings of the National Academy of Sciences 101 (2004): 5228–5235.
    doi:10.1073/pnas.0307752101.

    Wallach, Hanna, David Mimno, and Andrew McCallum. "Rethinking LDA: Why
    Priors Matter." In Advances in Neural Information Processing Systems 22,
    edited by Y.  Bengio, D. Schuurmans, J. Lafferty, C. K. I. Williams, and A.
    Culotta, 1973–1981, 2009.

    Buntine, Wray. "Estimating Likelihoods for Topic Models." In Advances in
    Machine Learning, First Asian Conference on Machine Learning (2009): 51–64.
    doi:10.1007/978-3-642-05224-8_6.

    """

    def __init__(self, n_topics, n_iter=2000, eta=0.01, sigma2=0.01, random_state=None,
                 refresh=10):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.eta = eta
        self.sigma2 = sigma2
        # if random_state is None, check_random_state(None) does nothing
        # other than return the current numpy RandomState
        self.random_state = random_state
        self.refresh = refresh

        # random numbers that are reused
        rng = check_random_state(random_state)
        self._rands = rng.rand(1024**2 // 8)  # 1MiB of random variates

        # configure console logging if not already configured
        if len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.NullHandler):
            logging.basicConfig(level=logging.INFO)

    def fit(self, X, sample_X, weights, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X, sample_X, weights)
        return self

    def fit_transform(self, X, sample_X, y=None):
        """Apply dimensionality reduction on X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions

        """
        if isinstance(X, np.ndarray):
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
            X = np.atleast_2d(X)
        self._fit(X, sample_X)
        return self.doc_topic_

    def transform(self, X, max_iter=20, tol=1e-16):
        """Transform the data X according to previously fitted model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        max_iter : int, optional
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double, optional
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions

        Note
        ----
        This uses the "iterated pseudo-counts" approach described
        in Wallach et al. (2009) and discussed in Buntine (2009).

        """
        if isinstance(X, np.ndarray):
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
            X = np.atleast_2d(X)
        doc_topic = np.empty((X.shape[0], self.n_topics))
        WS, DS = matrix_to_lists(X)
        # TODO: this loop is parallelizable
        for d in np.unique(DS):
            doc_topic[d] = self._transform_single(WS[DS == d], max_iter, tol)
        return doc_topic

    def _transform_single(self, doc, max_iter, tol):
        """Transform a single document according to the previously fit model

        Parameters
        ----------
        X : 1D numpy array of integers
            Each element represents a word in the document
        max_iter : int
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : 1D numpy array of length n_topics
            Point estimate of the topic distributions for document

        Note
        ----

        See Note in `transform` documentation.

        """
        PZS = np.zeros((len(doc), self.n_topics))
        for iteration in range(max_iter + 1): # +1 is for initialization
            PZS_new = self.components_[:, doc].T
            PZS_new *= (PZS.sum(axis=0) - PZS + self.alpha)
            PZS_new /= PZS_new.sum(axis=1)[:, np.newaxis] # vector to single column matrix
            delta_naive = np.abs(PZS_new - PZS).sum()
            logger.debug('transform iter {}, delta {}'.format(iteration, delta_naive))
            PZS = PZS_new
            if delta_naive < tol:
                break
        theta_doc = PZS.sum(axis=0) / PZS.sum()
        assert len(theta_doc) == self.n_topics
        assert theta_doc.shape == (self.n_topics,)
        return theta_doc

    def _fit(self, X, sample_X, weights):
        """Fit the model to the data X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features. Sparse matrix allowed.
        """
        random_state = check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize(X, weights)
        lambda_turning = LambdaTurning(self.n_topics, sample_X, self.ndz_, self.sigma2)
        self.lamb = np.random.randn(self.n_topics, sample_X.shape[1]) 
        self.sample_X = sample_X
        self.lamb = np.random.randn(self.n_topics, sample_X.shape[1]) * 0.1
        for it in range(self.n_iter):
            # FIXME: using numpy.roll with a random shift might be faster
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                ll = self.loglikelihood()
                logger.info("<{}> log likelihood: {:.0f}".format(it, ll))
                print("<{}> log likelihood: {:.0f}".format(it, ll))
                # keep track of loglikelihoods for monitoring convergence
                self.loglikelihoods_.append(ll)
            self._sample_topics(rands, weights)
            
            self.lamb = self.lamb.reshape((self.n_topics * sample_X.shape[1], )) + np.random.randn(self.n_topics * sample_X.shape[1]) * 0.1
            for i in range(100):
                grad = lambda_turning.gradLambda(self.lamb)
                self.lamb += 0.1 * grad
                self.lamb = np.clip(self.lamb, -5., 5.)
                if (i + 1) % 10 == 0:
                    print("%d max grad"%(i + 1), np.max(np.abs(grad)))
                
            #t_lamb = fmin_cg(lambda_turning.gradLambda, np.random.randn(self.n_topics * sample_X.shape[1]))
            self.lamb = self.lamb.reshape(self.n_topics, -1)
            
        ll = self.loglikelihood()
        logger.info("<{}> log likelihood: {:.0f}".format(self.n_iter - 1, ll))
        # note: numpy /= is integer division
        
        self.components_ = (self.nzw_ + self.eta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_
        self.doc_topic_ = (self.ndz_ + np.exp(np.dot(sample_X, self.lamb.T))).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        return self

    def _initialize(self, X, weights):
        D, W = X.shape
        N = int(X.sum())
        n_topics = self.n_topics
        n_iter = self.n_iter
        logger.info("n_documents: {}".format(D))
        logger.info("vocab_size: {}".format(W))
        logger.info("n_words: {}".format(N))
        logger.info("n_topics: {}".format(n_topics))
        logger.info("n_iter: {}".format(n_iter))

        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=float)
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=float)
        self.nz_ = nz_ = np.zeros(n_topics, dtype=float)

        self.WS, self.DS = WS, DS = matrix_to_lists(X)
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)
        np.testing.assert_equal(N, len(WS))
        for i in range(N):
            w, d = WS[i], DS[i]
            z_new = i % n_topics
            ZS[i] = z_new
            ndz_[d, z_new] += weights[d]
            nzw_[z_new, w] += weights[d]
            nz_[z_new] += weights[d]
        self.loglikelihoods_ = []

    def loglikelihood(self):
        """Calculate complete log likelihood, log p(w,z)

        Formula used is log p(w,z) = log p(w|z) + log p(z)
        """
        nzw, ndz, nz = self.nzw_, self.ndz_, self.nz_
        eta = self.eta
        nd = np.sum(ndz, axis=1).astype(float)
        return _lda._loglikelihood(nzw, ndz, nz, nd, eta, self.sample_X, self.lamb)

    def _sample_topics(self, rands, weights):
        """Samples all topic assignments. Called once per iteration."""
        n_topics, vocab_size = self.nzw_.shape
        eta = np.repeat(self.eta, vocab_size).astype(np.float64)
        _lda._sample_topics(self.WS, self.DS, self.ZS, self.nzw_, self.ndz_, self.nz_,
                                eta, rands, self.sample_X, self.lamb, weights)
