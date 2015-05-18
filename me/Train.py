__author__ = "Alexander Whillas <whillas@gmail.com>"


import time

"""
MaxEnt Markov Model training
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
import os

class Train(MachineLearningModule):
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.model = None

	def run(self, _):

		# Training the model
		print "Training MaxEnt model..."

		# Get data

		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		training_data = data.tagged_sents(self.config('training_file'))
		# cv_data = data.tagged_sents(self.config('cross_validation_file'))

		# Learn features

		self.model = MaxEntMarkovModel(Ratnaparkhi96Features, CollinsNormalisation)
		self.model.train(training_data)
		print "Features:", len(self.model.weights)

		# Learn feature weights

		# TODO Use the CV training set to tune the regularization_parameter of the MaxEntMarkovModel i.e. smaller param. learning cycles

		iterations = int(self.config('iterations'))
		reg = float(self.config('iterations'))
		mxitr = int(self.config('iterations'))
		self.load(filename_prefix="_params")
		for i in range(0, iterations):
			print "Iteration set #", i+1, "of", iterations  # incrementally... in case we overheat and crash :-/
			self.model.learn_parameters(training_data, regularization=reg, maxiter=mxitr)
			time.sleep(5)
			self.save(filename_prefix="_params,iter-{0},reg-{1},maxiter-{2}".format(i, reg, mxitr))

		# TODO: Use cross-validation set to tune the regularization param.

		return True
