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

        best_score = float("inf")
        best_model = None

        '''
        Description of parameters for BIC calculation below is taken from:

        http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf

        BIC = -2 * logL + p * logN
        where L is the likelihood of the fitted model, p is the number of parameters,
        and N is the number of data points.
        '''
        p = len(self.X[0])
        N = np.sum(self.lengths)
        logN = np.log(N)

        for comp_count in range(self.min_n_components, self.max_n_components + 1):

            try:
                model = self.base_model(comp_count)
                logL = model.score(self.X, self.lengths)

                # calc bic score
                bic_score = -2 * logL + p * logN

                if bic_score < best_score:
                    best_score = bic_score
                    best_model = model

            except:
                pass

        return best_model




class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    '''
    The purpose of this formula is that the:
    "Discriminant Factor Criterion is the difference between the evidence of the
    model, given the corresponding data set, and the average over anti-evidences of the model"

    - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
      section 3.1

    This helps to understand the formula, so we should score the log likelihood of the word, save that and
    then score the other words and save that too. Then we take the mean of the scores for the words that are
    the target word and and subtract that from the target word score.
    The model for the target word, that has the highest score (i.e. the highest difference between the mean for
    other words and itself) will be returned.
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_dic_score = float("-inf")

        other_words = [word for word in self.words if word != self.this_word]

        for num_states in range(self.min_n_components, self.max_n_components + 1):

            base_model = self.base_model(num_states)
            try:
                target_word_score = base_model.score(self.X, self.lengths)

                scores = list()

                for word in other_words:
                    word_x, word_length = self.hwords[word]
                    try:
                        scores.append(base_model.score(word_x, word_length))
                    except:
                        pass

                scores_mean = np.mean(scores)

                dic_score = target_word_score - scores_mean

                if dic_score > best_dic_score:
                    best_model = base_model

            except:
                pass

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    '''
    Number of splits is set to max 3 due to very long training time on words with many recordings.
    If set higher it has heavy impact on training time for some words.
    '''
    MAX_NUMBER_OF_SPLITS = 3
    MINIMUM_REQUIRED_SPLITS = 2

    def select(self):

        best_score = float("-inf")
        best_model = None
        log_l = float("-inf")

        for comp_count in range(self.min_n_components, self.max_n_components + 1):

            num_splits = min(self.MAX_NUMBER_OF_SPLITS, len(self.sequences))

            if num_splits >= self.MINIMUM_REQUIRED_SPLITS:
                split_method = KFold(n_splits=num_splits)

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                    self.X, self.lengths = combine_sequences(split_index_list=cv_train_idx, sequences=self.sequences)
                    x_test, x_test_lengths = combine_sequences(split_index_list=cv_test_idx, sequences=self.sequences)

                    try:
                        model = self.base_model(comp_count)
                        log_l = model.score(x_test, x_test_lengths)
                    except Exception as e:
                        pass

                    if log_l > best_score:
                        best_score = log_l
                        best_model = model

        return best_model