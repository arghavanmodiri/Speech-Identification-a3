from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch, sys
import random
import math
import time
from scipy.special import logsumexp

dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))  # omega is w for weights
        self.mu = np.zeros((M,d))  # means
        self.Sigma = np.zeros((M,d))  # covariance matrix

def precomputeM_1002971643(m, myTheta):

    mu_row = myTheta.mu[m]
    covar_row = myTheta.Sigma[m]
    d = len(mu_row)
    term_1 = np.sum(np.square(mu_row) / (2 * covar_row))
    term_2 = np.log(2*math.pi) * d / 2
    term_3 = np.log(np.product(covar_row)) / 2

    return np.sum([term_1, term_2, term_3])


def compute_intermediate_results_1002971643(M, X, myTheta):

    T = len(X)
    # Create a list of each of the m values from 0 to M-1 (inclusive)
    m_vals = [m for m in range(M)]
    # Pre-compute the values to be used in the log_b_m_x calculation
    precomputed_Ms = [precomputeM_1002971643(m, myTheta) for m in m_vals]

    # we must calculate all of the log_bs first so that we may use them in log_p_m_x_vectorized
    log_bs = np.empty((0, T))
    for m in m_vals:
        log_b_row = log_b_m_x_vectorized(m, X, myTheta, [precomputed_Ms[m]])
        log_bs = np.vstack([log_bs, log_b_row])

    # log_p_m_x_vectorized uses each of the T column vectors in its calculations
    log_ps = np.empty((0, T))
    log_p_x_theta = np.empty((0, T))
    for m in m_vals:
        log_p_row, log_p_denom = log_p_m_x_vectorized(m, myTheta, log_bs)
        log_ps = np.vstack([log_ps, log_p_row])
        if m == 0:
            log_p_x_theta = np.vstack([log_p_x_theta, log_p_denom])


    return log_bs, log_ps, log_p_x_theta

def update_paramaters_1002971643(myTheta, X, log_Ps, L):

    T = log_Ps.shape[1]
    sum_p_per_m = (math.e ** logsumexp(log_Ps, axis=1))
    omega_est = sum_p_per_m / T


    Ps = math.e**log_Ps
    mu_intermediate = np.dot(Ps, X)  # numerator of estimator
    mu_est = mu_intermediate / sum_p_per_m[:, None]  # numerator divided by the respective sum value


    x_squared = np.square(X)
    mu_est_squared = np.square(mu_est)
    sigma_intermediate = np.dot(Ps, x_squared)  # numerator of estimator
    sigma_intermediate_2 = sigma_intermediate / sum_p_per_m[:, None]  # numerator divided by the respective sum value
    sigma_est = sigma_intermediate_2 - mu_est_squared  # subtract each row by the respective mu_est values

    return omega_est, mu_est, sigma_est

def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout
        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''
    mu_row = myTheta.mu[m]
    covar_row = myTheta.Sigma[m]
    # we see there is a shared term in the two components of non-pre-computed part (x[n]/var[n])
    shared_term = np.divide(x, covar_row)
    term_1 = np.multiply(shared_term, x) / 2
    term_2 = np.multiply(shared_term, mu_row)
    term = np.sum(term_1 - term_2)

    independent_term = precomputeM_1002971643(m, myTheta)

    return -(term + independent_term)


def log_b_m_x_vectorized( m, X, myTheta, preComputedForM=[]):
    ''' Returns a 1xT vector of the log probabilities of each of the T rows
    of Txd matrix X using only component m of model myTheta.
    '''

    T = len(X)
    mu_row = myTheta.mu[m]
    covar_row = myTheta.Sigma[m]
    term_1 = (np.divide(np.square(X), covar_row)) / 2
    term_2 = np.divide(np.multiply(X, mu_row), covar_row)
    combined = np.subtract(term_1, term_2)
    combined_summed = np.sum(combined, axis=1).reshape((1, T))


    return -(combined_summed + preComputedForM[0])


def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''

    M = myTheta.mu.shape[0]
    denoms = []
    for i in range(M):
        log_bmx_val = log_b_m_x(i, x, myTheta)
        denoms.append(log_bmx_val)
    denoms = np.asarray(denoms).reshape((1, M))
    denoms = np.add(denoms, np.log(myTheta.omega.reshape((1, M))))
    numerator = denoms[0][m]
    denom = logsumexp(denoms)
    return np.subtract(numerator, denom)


def log_p_m_x_vectorized(m, myTheta, log_bs):
    ''' Returns the log probability of the each component for all time frames using log_bs previously computed.
    '''

    # Convert log_bs to a TxM matrix that details the log_b values across mixture components for each time frame
    components_per_frame = log_bs.T
    T = len(components_per_frame)
    logged_weights = np.log(myTheta.omega)
    weighed_values = np.add(components_per_frame, logged_weights)

    # term_1 is the value for each time frame for mixture component m
    term_1 = weighed_values[:, m].reshape((1, T))
    # term_2 is the sum of weighted values for each time frame
    term_2 = logsumexp(weighed_values, axis=1).reshape((1, T))
    pmx_vals = np.subtract(term_1, term_2)
    # we return term_2 to use in the likelihood function
    return pmx_vals, term_2

    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x
        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).
        We don't actually pass X directly to the function because we instead pass:
        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.
        See equation 3 of the handout
    '''
    components_per_frame = log_Bs.T
    T = len(components_per_frame)
    M = len(components_per_frame[0])
    omegas = myTheta.omega.reshape((1, M))
    logged_weights = np.log(omegas)
    weighed_values = np.add(components_per_frame, logged_weights)

    # liks is the sum of weighted values for each time frame
    liks = logsumexp(weighed_values, axis=1).reshape((1, T))
    return np.sum(liks)
    


def logLik_vectorized( log_p_x_thetas):
    ''' Return the log likelihood of 'X' using precomputed log(p_x_thetas)
    '''

    # Since we have already computed the log(p(x;theta)) values for each time frame, we can just sum them
    return np.sum(log_p_x_thetas)


def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''

    T = X.shape[0]
    d = X.shape[1]
    myTheta = theta(speaker, M, d)

    # Re-initialize theta parameters correctly
    myTheta.omega = np.repeat(1 / M, M).reshape((1, M))
    myTheta.Sigma = np.ones((M, d))
    init_mu_indices = np.random.randint(T, size=M)
    myTheta.mu = X[init_mu_indices, :]

    i = 0
    prev_L = float("-inf")
    improvement = float("inf")
    while i <= maxIter and improvement >= epsilon:
        log_Bs, log_Ps, log_p_x_thetas = compute_intermediate_results_1002971643(M, X, myTheta)

        L = logLik_vectorized(log_p_x_thetas)
        myTheta.omega, myTheta.mu, myTheta.Sigma = update_paramaters_1002971643(myTheta, X,  log_Ps, L)
        improvement = L - prev_L
        prev_L = L
        i += 1

    myTheta.omega = myTheta.omega.reshape((1, M))
    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 
        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    # test( testMFCCs[i], i, trainThetas, k )
    # testMFCCs[i] is X so its Txd
    # i is an index for number of testMfccs
    # trainthetas are all the trained models
    # k is number of features
    T = mfcc.shape[0]
    d = mfcc.shape[1]
    M = models[0].omega.shape[1]
    print(models[i].name)
    model_to_lik = {}
    likelihoods = []
    for model in models:
        log_Bs, log_ps, log_p_thetas = compute_intermediate_results_1002971643(M, mfcc, model)
        l_likelihood = logLik_vectorized(log_p_thetas)
        model_to_lik[l_likelihood] = model.name
        likelihoods.append(l_likelihood)
    max_liks = sorted(model_to_lik.keys(), reverse=True)[:k]
    for lik in max_liks:
        print("{} {}".format(model_to_lik[lik], lik))
    bestModel = np.argmax(likelihoods)
    return 1 if (bestModel == correctID) else 0



if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20

    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )

            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )
            #trainThetas.append(theta("help"))

    # evaluate
    sys.stdout = open("gmmLiks.txt", "w")
    numCorrect = 0
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k )
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print(accuracy)
