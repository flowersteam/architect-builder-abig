import numpy as np
from scipy.special import log_softmax, softmax, logsumexp


class SoftValueIteration(object):
    def __init__(self, nS, nA, discount_factor, threshold=0.0001, temperature=0.01, mode="normal"):
        self.alpha = temperature
        self.mode = mode
        self.threshold = threshold
        self.discount_factor = discount_factor
        self.nS, self.nA = nS, nA

    def learn(self, Pmat, Rmat):
        # Directly returns the Q-values times the TEMPERATURE
        # Pmat.shape = Rmat.shape = (nS, nA, nS)

        Qmat = np.zeros((self.nS, self.nA))
        old_V = None
        i = 0

        while True:

            if self.mode == 'normal':
                policy = softmax(Qmat / self.alpha, axis=1)
                log_policy = log_softmax(Qmat / self.alpha, axis=1)
                V = np.einsum('sa->s', policy * (Qmat - self.alpha * log_policy))

            elif self.mode == 'simplified':
                # just a fancier expression that you find by develloping the above formula analytically
                V = self.alpha * logsumexp(1 / self.alpha * Qmat, axis=1)

            else:
                raise NotImplementedError

            Qmat = np.einsum('san->sa', Pmat * (Rmat + self.discount_factor * V.reshape((1, 1, -1))))

            if old_V is not None:
                delta = np.einsum('s->', (V - old_V) ** 2) ** 0.5
            else:
                delta = np.inf

            if delta <= self.threshold:
                break
            else:
                i += 1
                old_V = V

        return Qmat / self.alpha


class FactoredSoftValueIteration(object):
    def __init__(self, nS, nM, nA, discount_factor, message_model_type, threshold=0.0001, temperature=0.01,
                 max_iter=10000):
        self.alpha = temperature
        self.threshold = threshold
        self.discount_factor = discount_factor
        self.message_model_type = message_model_type
        self.nS, self.nM, self.nA = nS, nM, nA
        self.max_iter = max_iter

    def learn(self, Pmat_env, Rmat):
        # Directly returns the Q-values times the TEMPERATURE
        # Pmat_env.shape = (nS, nA, nS)
        # Rmat.shape = (nS, nM, nA, nS, nM), the irl reward matrix

        Qmat = np.zeros((self.nS, self.nM, self.nA))
        old_V = None
        i = 0

        while True:

            policy = softmax(Qmat / self.alpha, axis=2)  # policy.shape = (nS, nM, nA)
            log_policy = log_softmax(Qmat / self.alpha, axis=2)  # log_policy.shape = (nS, nM, nA)

            # We compute the message transition probability matrix assuming "shared intent": softmax on the expected
            # message-utility
            V = np.einsum('sma,sma->sm', policy, (Qmat - self.alpha * log_policy))  # V.shape = (nS, nM)

            if self.message_model_type == 'softmax':
                Pmat_sm = softmax(V / self.alpha, axis=1)  # note that we use the same alpha-temperature
            elif self.message_model_type == 'uniform':
                Pmat_sm = np.ones((self.nS, self.nM)) / self.nM
            else:
                raise NotImplementedError
                # The full builder-env transition matrix can be factored into the env transition probabilities
            # and the message transition probabilities
            Pmat = np.einsum('san, nk -> sank', Pmat_env, Pmat_sm)  # Pmat.shape = (nS, nA, nS, nM)

            #  we have to reshape V so that it broadcast well with Rmat that is (nS, nM, nA, nS, nM)
            Qmat = np.einsum('sank,smank->sma', Pmat,
                             Rmat + self.discount_factor * V[np.newaxis, np.newaxis, np.newaxis, :, :])

            if old_V is not None:
                delta = np.einsum('sm->', (V - old_V) ** 2) ** 0.5
            else:
                delta = np.inf

            if delta <= self.threshold or i >= self.max_iter:
                break
            else:
                i += 1
                old_V = V

        return Qmat / self.alpha

    def build_Rmat_from_reward_fct(self, reward_fct):
        # only ok for DETERMINISTIC REWARD FCT
        # reward function is such that r = reward_fct(s,m,a,s',m')

        Rmat = np.zeros((self.nS, self.nM, self.nA, self.nS, self.nM))

        for s in range(self.nS):
            for m in range(self.nM):
                for a in range(self.nA):
                    for sp in range(self.nS):
                        for mp in range(self.nM):
                            Rmat[s, m, a, sp, mp] = reward_fct(s, m, a, sp, mp)

        return Rmat
