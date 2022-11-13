# Intro to AI and AI Applications, Kookmin Univ
# HW4 - skeleton code 
# This code heavily borrows Assignment 3 of Prof. Rob Voigt's Ling334 course at Northwestern University
#  url: https://faculty.wcas.northwestern.edu/robvoigt/courses/2021_spring/ling334/assignments/a3.html


# *** imported packages *** #
#  (IT IS RECOMMENDED TO USE THESE PACKAGES IN YOUR CODE) #
import os, math
from collections import Counter

# *** NaiveBayesClassifier class *** #
class NaiveBayesClassifier:
    """Code for a bag-of-words Naive Bayes classifier.
    """
    
    def __init__(self, train_dir='haiti/train', REMOVE_STOPWORDS=False):
        """initialization of the Naive Bayes classification model class
           class attributes for training data and training results are initialized
        """
        # options related to training data
        # classes: list of class names (read from file names in training data root folder)
        self.classes = os.listdir(train_dir)  
        # train_data: list of file paths of training data files
        self.train_data = {c: os.path.join(train_dir, c) for c in self.classes}
        
        # variables for training results
        # vocabulary: set of unique words in training text data
        self.vocabulary = set([])
        # logprior: represents [appearance probability] of each class 
        self.logprior = {}  # log P(c) in algorithm
        # loglikelihood: represents [conditional probability] of each word for class
        #                keys should be tuples in the form (w, c)
        self.loglikelihood = {}  # log P(w|c) in algorithm

        # options related to use of STOPWORDS (words to be removed from text)
        self.stopwords = set([l.strip() for l in open('english.stop')])  # stopwords loaded from file

        
    def train(self):
        """Train the Naive Bayes classification model, following the pseudocode for
        training given in Figure 4.2 of the reference pdf file.)
        (Chapter 4 of Speech and Language Processing. Daniel Jurafsky & James H. Martin)

        Note that self.train_data contains the paths to training data files. 
        To get all the documents for a given training class c in a list, you can use:
            c_docs = open(self.train_data[c]).readlines()
        you can get words with: 
            words = doc.split()
        where [doc] is element of [c_docs]

        When converting from the pseudocode, consider how many loops over the data you
        will need to properly estimate the parameters of the model, and what intermediary
        variables you will need in this function to store the results of your computations.

        Parameters
        ----------
        None (reads training data from self.train_data)
        
        Returns
        -------
        None (updates class attributes self.vocabulary, self.logprior, self.loglikelihood)
        """
        # # >>> YOUR ANSWER <<< # #
        # YOU MUST:
        # 1. read training data files and store [all-words-per-class list] as a dictionary
        # 2. count and store number of 'documents' (lines) per-class (in each file) = N_c
        # 3. count and store total number of 'documents' (lines) in all files = N_doc
        # 4. compute dictionary self.logprior[c] 
        # 5. construct set self.vocabulary from dictionary of words-per-class bigdoc[c]
        #   (use the Python internal function set() on the list of all words to do this)
        # 6. compute dictionary self.loglikelihood[(w,c)] by 
        #  - count the [number of occurance of each word in vocabulary in bigdoc[c]]
        #   (use the Python internal function Counter() on bigdoc[c] to do this)
        #  - compute the [total sum of numbers of occurance of all words]
        #  - compute log of ratio of each occurance number to total sum for all words
        

        # # >>> END YOUR ANSWER <<< # #
  

    def train_stopwords(self):
        """Train the Naive Bayes classification model, following the pseudocode for
        training given in Figure 4.2 of the reference pdf file.)
        (Chapter 4 of Speech and Language Processing. Daniel Jurafsky & James H. Martin)

        In this function, you must take STOPWORDS into account;
        that is, self.stopwords should be removed whenever they appear.

        Parameters
        ----------
        None (reads training data from self.train_data)
        
        Returns
        -------
        None (updates class attributes self.vocabulary, self.logprior, self.loglikelihood)
        """
        # # >>> YOUR ANSWER <<< # #
        # YOU MUST: train just as train() function, 
        #           but with an additional process to remove stopwords from self.vocabulary

        
        # # >>> END YOUR ANSWER <<< # #


    def score(self, doc, c):
        """Return the log-probability of a given document for a given class,
        using the trained Naive Bayes classifier. 

        This is analogous to the inside of the for loop in the TestNaiveBayes
        pseudocode in Figure 4.2, SLP Chapter 4.

        Parameters
        ----------
        doc : str
            The text of a document to score.
        c : str
            The name of the class to score it against.

        Returns
        -------
        float
            The log-probability of the document under the model for class c.
        """        
        # # >>> YOUR ANSWER HERE <<< # #
        # YOU MUST: 
        # 1. for input doc, compute and return score for assumed class c
        
        # # >>> END YOUR ANSWER <<< # #
                
    def predict(self, doc):
        """Return the most likely class for a given document under the trained classifier model.
        This should be only a few lines of code, and should make use of your self.score function.

        Consider using the `max` built-in function. There are a number of ways to do this:
           https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

        Parameters
        ----------
        doc : str
            A text representation of a document to score.
        
        Returns
        -------
        str
            The most likely class as predicted by the model.
        """
        # >>> YOUR ANSWER HERE
        # YOU MUST: 
        # 1. for input doc, compute score for all class ('relevant' and 'irrelevant')
        # 2. determine the class with the max score for input doc
        # 3. return the most likely class as predicted by the model
        
        # >>> END YOUR ANSWER


    def evaluate(self, test_dir='haiti/test', target='relevant'):
        """Calculate a precision, recall, and F1 score for the model
        on a given test set.

        Note the 'target' parameter here, giving the name of the class
        to calculate relative to. So you can consider a True Positive
        to be an instance where the gold label for the document is the
        target and the model also predicts that label; a False Positive
        to be an instance where the gold label is *not* the target, but
        the model predicts that it is; and so on.

        Parameters
        ----------
        test_dir : str
            The path to a directory containing the test data.
        target : str
            The name of the class to calculate relative to. 

        Returns
        -------
        (float, float, float)
            The model's precision, recall, and F1 score relative to the
            target class.
        """        
        test_data = {c: os.path.join(test_dir, c) for c in self.classes}
        if not target in test_data:
            print('Error: target class does not exist in test data.')
            return
        outcomes = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        # >>> YOUR ANSWER HERE
        # 1. read test data files
        # 2. for all doc in data file, get predicted class
        # 3. if predicted class == target class --> result is [Positive]
        # 4. if predicted class == true class --> result is [True]
        #    classify doc prediction as 
        #    [TP=True & Positive] OR [TN=True & Negative] OR
        #    [FP=False & Positive] OR [FN=False & Negative]
        # 5. compute precision, recall, and F1 score 
        #    reference: https://en.wikipedia.org/wiki/Precision_and_recall
        
        precision = 0.0   # replace with equation for precision
        recall = 0.0      # replace with equation for recall
        f1_score = 0.0    # replace with equation for f1
        # >>> END YOUR ANSWER
        return (precision, recall, f1_score)


    def print_top_features(self, k=10):
        results = {c: {} for c in self.classes}
        for w in self.vocabulary:
            for c in self.classes:
                ratio = math.exp( self.loglikelihood[w, c] - min(self.loglikelihood[w, other_c] for other_c in self.classes if other_c != c) )
                results[c][w] = ratio

        for c in self.classes:
            print(f'Top features for class <{c.upper()}>')
            for w, ratio in sorted(results[c].items(), key = lambda x: x[1], reverse=True)[0:k]:
                print(f'\t{w}\t{ratio}')
            print('')
            



