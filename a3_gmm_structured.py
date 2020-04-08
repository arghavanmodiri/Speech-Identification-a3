from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import sys
import random

dataDir = "/u/cs401/A3/data/"


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        '''
        0.5 * (np.dot(self.mu[m]**2, 1.0/self.Sigma[m]) + d*np.log(2*np.pi) + np.log(np.prod(self.Sigma[m])))
        '''
        '''Vectorized
        d = self._d
        result = np.multiply(self.mu**2, 1.0/self.Sigma).sum(axis=1)
        result = result + d*np.log(2*np.pi)
        result = result + np.log(np.prod(self.Sigma, axis=1))
        result = result * -0.5
        '''
        d = self._d
        result = np.dot(self.mu[m]**2, 1.0/self.Sigma[m])
        result = result + d*np.log(2*np.pi)
        result = result + np.log(np.prod(self.Sigma[m]))
        result = result * -0.5
        return result

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    x_2 = x ** 2
    Sigma_reverse = 1.0/myTheta.Sigma[m]
    xSigma = np.multiply(x, Sigma_reverse)
    if np.isscalar(x[0]):
        result = -0.5 * np.dot(x_2, Sigma_reverse) + np.dot(myTheta.mu[m],xSigma)
    else:
        x2Sigma = np.multiply(x_2, Sigma_reverse)
        result = -0.5 * x2Sigma.sum(axis=1) + np.multiply(myTheta.mu[m], xSigma).sum(axis=1)

    result = result + myTheta.precomputedForM(m)
    return result


def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    log_omega = np.log(myTheta.omega)

    M = myTheta.mu.shape[0]

    log_b_omega = log_Bs + log_omega

    #log_b_omega = log_b_omega.T
    max_val = np.max(log_b_omega, axis=0, keepdims=True)

    result = max_val + np.log(np.sum(np.exp(log_b_omega - max_val), axis=0))
    
    result = log_b_omega - result
    return result


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    log_omega = np.log(myTheta.omega)
    print("log_omega: \n", log_omega)
    result = log_omega + log_Bs
    print("log_omega+B: \n", result)
    max_val = np.max(result, axis=0, keepdims=True)
    print("max_val: \n", max_val)
    print("np.exp(result - max_val): \n", np.exp(result - max_val))
    print("np.sum(np.exp(result - max_val): \n", np.sum(np.exp(result - max_val), axis=0))
    print("np.log(np.sum(np.exp(result - max_val): \n", np.log(np.sum(np.exp(result - max_val), axis=0)))
    result = max_val + np.log(np.sum(np.exp(result - max_val), axis=0))
    print("result ghable sum: \n", result)
    result = result.sum()
    print("result: \n", result)
    return result


def UpdateParameters(myTheta, X, log_Ps, L):
    T = X.shape[0] * 1.0
    Ps = np.exp(log_Ps)
    sum_Ps = Ps.sum(axis=-1).reshape(-1,1)
    new_omega=sum_Ps/T
    new_mu=np.matmul(Ps, X) / sum_Ps
    new_Sigma=np.matmul(Ps, X**2) / sum_Ps - new_mu**2

    myTheta.reset_omega(new_omega)
    myTheta.reset_mu(new_mu)
    myTheta.reset_Sigma(new_Sigma)
    '''
    new_omega = sum_Ps/T
    new_mu = np.matmul(Ps, X) / sum_Ps
    new_Sigma = np.matmul(Ps, X**2) / sum_Ps
    new_Sigma = new_Sigma - new_mu**2

    return [new_omega, new_mu, new_Sigma]
    '''

def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    # perform initialization (Slide 32)
    print("TODO : Initialization")
    # for ex.,
    T = X.shape[0]
    d = myTheta._d
    random_idx = np.random.randint(T, size=M)
    myTheta.reset_omega([1.0/M]*M)
    myTheta.reset_mu(X[random_idx, :])
    myTheta.reset_Sigma([[1.0/M]*d]*M)

    i=0
    prev_L = -np.inf
    improvement = np.inf

    while i<=maxIter and improvement>=epsilon:
        #ComputeIntermediateResults ;
        log_Bs = log_b_m_x(0, X, myTheta)
        for m in range(1, M):
            log_Bs = np.vstack([log_Bs, log_b_m_x(m, X, myTheta)])
        #log_Ps = log_p_m_x(0, X, myTheta)
        log_Ps = log_p_m_x(log_Bs, myTheta)
        #for m in range(1, M):
        #    log_Ps = np.vstack([log_Ps, log_p_m_x(m, X, myTheta)])

        L = logLik(log_Bs, myTheta)
        #, log_p_x_thetas = compute_intermediate_results_1002971643(M, X, myTheta)
        UpdateParameters(myTheta, X, log_Ps, L)
        #L := ComputeLikelihood (X, θ) ;
        #θ := UpdateParameters (θ, X, L) ;
        improvement = L - prev_L
        prev_L = L
        #print(i, " prev_L : ", prev_L)
        #print(i, " improvement : ", improvement)
        i += 1

    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    bestModel = -1
    all_logLik = []
    model_names = []
    model_to_logLik = {}
    for model in models:
        log_Bs = log_b_m_x(0, mfcc, model)
        for m in range(1, model._M):
            log_Bs = np.vstack([log_Bs, log_b_m_x(m, mfcc, model)])
        Ps = logLik(log_Bs, model)
        all_logLik.append(Ps)
        model_names.append(model.name)
        model_to_logLik[Ps] = model.name

    top_logLiks = sorted(model_to_logLik.keys(), reverse=True)[:k]

    #top_Ps = all_logLik[np.argpartition(all_logLik , len(all_logLik)-1)[-k:]]
    if k > 0:
        print(models[correctID].name)
        for lik in top_logLiks:
            print("{} {}".format(model_to_logLik[lik], lik))
    bestModel = np.argmax(all_logLik)


    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print("TODO: you will need to modify this main block for Sec 2.3")
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    sys.stdout = open("gmmLiks.txt", "w")
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print(accuracy)