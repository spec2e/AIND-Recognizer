import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # loop through all sequences with the index as key
    for index in test_set.get_all_sequences().keys():
        # obtain the data to score
        seq, x_length = test_set.get_item_Xlengths(index)

        # this is where we store the results for each word/model set with the obtained data
        word_log_l = dict()
        for word, model in models.items():
            try:
                log_l = model.score(seq, x_length)
            except Exception as e:
                # in case the scoring fails, set the log likelihood to negative infinity
                log_l = float("-inf")
                pass

            word_log_l[word] = log_l

        # append the dictionary with obtained probabilities to the list of total probabilities
        probabilities.append(word_log_l)

        # append the best result to the list of guesses
        guesses.append(max(word_log_l, key = word_log_l.get))

    return probabilities, guesses
