from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random

dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))
        self.save_log_Bs = [0] * M
        #self.preComputedTerms = self.preCompute(d)

    @property
    def preComputedForM(self):
        d = self.mu.shape[1]
        result = np.multiply(self.mu**2, 1.0/self.Sigma).sum(axis=1)
        result = result + d*np.log(2*np.pi)
        result = result + np.log(np.prod(self.Sigma, axis=1))
        result = result * -0.5
        '''
        0.5 * (np.dot(self.mu[m]**2, 1.0/self.Sigma[m]) + d*np.log(2*np.pi) + np.log(np.prod(self.Sigma[m])))
        '''
        return result


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    '''
    Returns the log probability of d-dimensional vector x using only
    component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    If you do this, you pass that precomputed component in preComputedForM
    '''
    #print("*********************** log_b_m_x ***********************")
    #print(x)

    x_2 = x ** 2
    Sigma_reverse = 1.0/myTheta.Sigma[m]
    xSigma = np.multiply(x, Sigma_reverse)
    if np.isscalar(x[0]):
        result = np.dot(x_2, Sigma_reverse) + np.dot(myTheta.mu[m],xSigma)
    else:
        x2Sigma = np.multiply(x_2, Sigma_reverse)
        result = x2Sigma.sum(axis=1) + np.multiply(myTheta.mu[m], xSigma).sum(axis=1)
    result = -0.5 * result + myTheta.preComputedForM[m]

    #print("result")
    #print(result)
    myTheta.save_log_Bs[m] = result
    return result

def log_p_m_x(m, x, myTheta):
    '''
    Returns the log probability of the m^{th} component given d-dimensional
    vector x, and model myTheta (See equation 2 of handout)
    '''
    #print("*********************!! log_p_m_x !!*********************")
    if np.isscalar(x[0]):
        axis=-1
    else:
        axis=1
    log_omega=np.log(myTheta.omega)

    M = myTheta.mu.shape[0]
    if myTheta.save_log_Bs==None:
        log_b_omega = log_b_m_x(0, x, myTheta) + log_omega[0]

        for i in range(1, M):
            log_b_omega = np.vstack((log_b_omega, log_b_m_x(i, x, myTheta) + log_omega[i]))
    else:
        save_log_Bs = np.vstack(myTheta.save_log_Bs)
        #print("log_omega ", log_omega.shape)
        #print("save_log_Bs ", save_log_Bs.shape)
        log_b_omega = save_log_Bs + log_omega
        #print("log_b_omega ", log_b_omega.shape)

    #log_b_omega = log_b_omega.T
    max_val = np.max(log_b_omega, axis=0, keepdims=True)
    #print("max_val ", max_val.shape)
    #print("np.exp(log_b_omega - max_val) ", np.exp(log_b_omega - max_val).shape)

    result = max_val + np.log(np.sum(np.exp(log_b_omega - max_val), axis=0, keepdims=True))
    
    #print("result ", result.shape)
    #print("log_b_omega[m] ", log_b_omega[m].shape)
    result = log_b_omega[m] - result
    result = result.reshape((-1,))
    #print("result ", result.shape)
    #print(result)
    #print("^^^^^^^^^^^^")
    return result

    
def logLik(log_Bs, myTheta):
    '''
    Return the log likelihood of 'X' using model 'myTheta' and precomputed
    MxT matrix, 'log_Bs', of log_b_m_x
    X can be training data, when used in train( ... ), and
    X can be testing data, when used in test( ... ).
    We don't actually pass X directly to the function because we instead pass:
    log_Bs(m,t) is the log probability of vector x_t in component m, which is
    computed and stored outside of this function for efficiency. 
    See equation 3 of the handout
    '''
    result = myTheta.omega + log_Bs
    #print("*********** result ", result.shape)
    max_val = np.max(result, axis=0, keepdims=True)
    result = max_val + np.log(np.sum(np.exp(result - max_val), axis=0))
    #print("*********** result ", result.shape)
    result = result.sum()
    #print("*********** result ", result.shape)
    #print("*********** result ", result)
    return result

def UpdateParameters(myTheta, X, log_Ps, L):
    T = X.shape[0]
    Ps = np.exp(log_Ps)
    sum_Ps = Ps.sum(axis=1).reshape(-1,1)
    new_omega = sum_Ps/T
    print("new_omega : ", new_omega.shape)

    new_mu = np.matmul(Ps, X) / sum_Ps

    new_Sigma = np.matmul(Ps, X**2) / sum_Ps
    new_Sigma = new_Sigma - new_mu**2

    return [new_omega, new_mu, new_Sigma]

    #new_omega


def train(speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    '''print(type(X))
    print(X)
    print(X.shape)
    myTheta = theta(speaker, M, X.shape[1])
    #log_p_m_x(7, X[0], myTheta)
    b0 = log_b_m_x(0, X[0:2], myTheta)
    print("b0.shape : ", log_p_m_x(0, X[0:3], myTheta).shape)
    b1 = log_b_m_x(1, X[0:2], myTheta)
    b2 = log_b_m_x(2, X[0:2], myTheta)
    b3 = log_b_m_x(3, X[0:2], myTheta)
    b4 = log_b_m_x(4, X[0:2], myTheta)
    b5 = log_b_m_x(5, X[0:2], myTheta)
    b6 = log_b_m_x(6, X[0:2], myTheta)
    b7 = log_b_m_x(7, X[0:2], myTheta)
    b01=np.vstack((b0,b1))
    b012=np.vstack((b01, b2))
    b0123=np.vstack((b012, b3))
    b01234=np.vstack((b0123, b4))
    b012345=np.vstack((b01234, b5))
    b0123456=np.vstack((b012345, b6))
    B = np.vstack((b0123456,b7))
    logLik(B, myTheta)'''
    myTheta = theta(speaker, M, X.shape[1])
    T = X.shape[0]
    d = X.shape[1]
    random_idx = np.random.randint(T, size=M)
    myTheta.omega.fill(1.0/M)
    myTheta.Sigma.fill(1.0/M)
    myTheta.mu = X[random_idx, :]
    print("in TRAIN")

    i=0
    prev_L = -np.inf
    improvement = np.inf

    while i<=maxIter and improvement>=epsilon:
        #ComputeIntermediateResults ;
        log_Bs = log_b_m_x(0, X, myTheta)
        for m in range(1, M):
            log_Bs = np.vstack([log_Bs, log_b_m_x(m, X, myTheta)])

        log_Ps = log_p_m_x(0, X, myTheta)
        for m in range(1, M):
            log_Ps = np.vstack([log_Ps, log_p_m_x(m, X, myTheta)])

        L = logLik(log_Bs, myTheta)
        #, log_p_x_thetas = compute_intermediate_results_1002971643(M, X, myTheta)
        [myTheta.omega, myTheta.mu, myTheta.Sigma] = UpdateParameters(myTheta, X, log_Ps, L)

        #L := ComputeLikelihood (X, θ) ;
        #θ := UpdateParameters (θ, X, L) ;
        improvement = L - prev_L
        prev_L = L
        i += 1

    return myTheta


def test(mfcc, correctID, models, k=5 ):
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
    bestModel = -1
    print ('TODO')
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        #print("**********************")
        #print(subdir)
        #print(dirs)
        #print(files)
        for speaker in dirs:
            #print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)),
                '*npy')
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load(os.path.join( dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate 
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0*numCorrect/len(testMFCCs)

