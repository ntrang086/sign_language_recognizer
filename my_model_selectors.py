import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_score = float("INF")
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)

                # Calculate the number of free parameters, 
                # based on https://en.wikipedia.org/wiki/Hidden_Markov_model
                # and https://discussions.udacity.com/t/verifing-bic-calculation/246165/6
                
                # Parameters for initial probability distribution
                initial_prob_params = n-1 # Subtract 1 because an array of probabilities 
                                          # must add up to 1 and
                # Transition parameters
                transition_params = n * (n-1) # Subtract 1 for the same reason as above

                num_features = len(model.means_[0])

                # Emission parameters = num_mean_params + num_covar_params                
                # Since covariance_type == "diag", num_covar_params == num_mean_params
                emission_params = 2*n*num_features

                p = initial_prob_params + transition_params + emission_params
                
                score = -2*logL + p * math.log(len(self.X))
                if best_score > score:
                    best_score = score
                    best_model = model
                
                if self.verbose:
                    print("model created for {} with {} states".format(self.this_word, num_states))     
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
        return best_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_score = float("-INF")
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                
                # Compute the anti-likelihood for other words
                anti_logL = []
                for other_word in self.hwords:
                    if other_word != self.this_word:
                        other_X, other_lengths = self.hwords[other_word]
                        anti_logL.append(model.score(other_X, other_lengths))
                
                score = logL - sum(anti_logL)/(len(self.hwords) - 1)
                if best_score < score:
                    best_score = score
                    best_model = model
                
                if self.verbose:
                    print("model created for {} with {} states".format(self.this_word, num_states))     
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
        return best_model        


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = float("-INF")
        best_model = None
        try:
            if len(self.lengths) >= 2:
                split_method = KFold(n_splits=min(3,len(self.lengths)))
                for n in range(self.min_n_components, self.max_n_components+1):
                    scores = []
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        scores.append(model.score(X_test, lengths_test))

                    score = np.mean(scores)
                    if best_score < score:
                        best_score = score
                        best_model = self.base_model(n)
            else:
                best_model = self.base_model(self.n_constant)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))     
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))                
        return best_model
