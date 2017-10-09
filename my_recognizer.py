import warnings
import arpa
import os
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
    # TODO implement the recognizer
    test_dict = test_set.get_all_Xlengths()
    for i in range(test_set.num_items):
        # Get the feature lists and lengths for the current id
        X, lengths = test_dict[i]
        prob_dict = {}
        for word in models:
            try:
                # Calculate the logL for each word based on the corresponding model
                prob_dict[word] = models[word].score(X, lengths)
            except:
                #print("failure on index {} and word {}".format(i, word))
                prob_dict[word] = float("-INF")
                pass
        probabilities.append(prob_dict)
        # Add the word with the highest logL into guesses
        guesses.append(max(prob_dict, key=lambda key: prob_dict[key]))
    return probabilities, guesses
    


# Improve WER with language models
# Data from ftp://wasserstoff.informatik.rwth-aachen.de/pub/rwth-boston-104/lm/
# Explanation of n-gram format: http://www.speech.sri.com/projects/srilm/manpages/ngram-format.5.html
def recognize_lm(models: dict, test_set: SinglesData, scaling_factor=15.0):
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

    # TODO implement the recognizer
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # Load the ngram files
    unigram = os.path.join('slm_data', 'devel_lm_M1_sri.lm')
    bigram = os.path.join('slm_data', 'devel_lm_M2_sri.lm')
    trigram = os.path.join('slm_data', 'devel_lm_M3_sri.lm')

    # Source code for arpa: https://github.com/sfischer13/python-arpa/blob/master/arpa/models/base.py
    language_model = arpa.loadf(trigram)[0]

    """
    # Example of the relation between log_s and log_p
    print ("log_s and log_p for THROW APPLE WHO")
    print ("log_s without sentence markers", language_model.log_s("THROW APPLE WHO", sos=False, eos=False))
    print ("log_s with sentence markers", language_model.log_s("THROW APPLE WHO"))
    print ("log_p of the whole sentence", language_model.log_p("THROW APPLE WHO"))
    print ("log_p of the first word", language_model.log_p("THROW"))
    print ("log_p of the second word", language_model.log_p("APPLE"))
    print ("log_p of the third word", language_model.log_p("WHO"))
    print ("log_p of the first and second word", language_model.log_p("THROW APPLE"))
    print ("log_p of the whole sentence + log_p of first & second + log_p of the first word ", 
            language_model.log_p("THROW APPLE WHO") + language_model.log_p("THROW APPLE") 
            + language_model.log_p("THROW")) # == log_s without sentence markers 
    """

    # Load all video sentence numbers from the test_set
    sentences_index = test_set.sentences_index
    #print (sentences_index)
    #{2: [0, 1, 2], 7: [3, 4, 5, 6], ..., 28: [24, 25, 26, 27, 28], ...}

    # Load all feature lists and their lengths from the test_set
    test_dict = test_set.get_all_Xlengths()
    for video_id in sentences_index:
        word_ids = sentences_index[video_id]
        # A list of dictionaries of top predictions & their logL for each word
        top_probs = []
        for word_id in word_ids:
            X, lengths = test_dict[word_id]
            prob_dict = {}
            for word in models:
                try:
                    # Calculate the logL for each word based on the corresponding model
                    prob_dict[word] = models[word].score(X, lengths)
                except:
                    #print("failure on index {} and word {}".format(word_id, word))
                    prob_dict[word] = float("-INF")

            probabilities.append(prob_dict)
            # Top 3 yields a better WER than other options
            top_words = sorted(prob_dict, key=prob_dict.get, reverse=True)[:3]
            top_probs.append({word:prob_dict[word] for word in top_words})

        # Create a list of possible sentences from the top predictions
        sentences = product(*top_probs)
        best_sentence_score = float("-INF")
        best_sentence = []

        # Calculate each sentence's score, a combination of word & language model scores
        # based on: https://www-i6.informatik.rwth-aachen.de/publications/download/154/Dreuw--2007.pdf
        # Then pick the sentence with the highest score
        for sentence in sentences:
            word_model_prob = 0 # Based on the HMM
            for i in range(len(sentence)):
                word_model_prob = word_model_prob + top_probs[i][sentence[i]]

            sentence_str = (" ".join(word for word in sentence)).strip()
            try:
                language_model_prob = language_model.log_s(sentence_str)
                sentence_score = scaling_factor*language_model_prob + word_model_prob
            except:
                sentence_score = float("-INF")
            # <= makes sure that best_sentence is assigned at least a value
            # for example, when sentence_score is "-INF" for all possible sentences 
            # of a video_id, best_sentence will have "-INF" value instead of []
            if best_sentence_score <= sentence_score: 
                best_sentence_score = sentence_score
                best_sentence = sentence
        guesses.extend(best_sentence)
    return probabilities, guesses

# based on itertools.product() https://docs.python.org/3/library/itertools.html#itertools.product
# but it returns a list of lists instead of a list of tuples
def product(*args):
    pools = [tuple(pool) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

