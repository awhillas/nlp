__author__ = "Alexander Whillas <whillas@gmail.com>"

"""
MaxEnt Markov Model training
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, HonnibalFeats, CollinsNormalisation
import os

class Train(MachineLearningModule):
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.model = None

	def run(self, _):

		# Training the model

		print "Training MaxEnt model..."
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		data_file = self.config('training_file')  # use smaller set for development
		self.model = MaxEntMarkovModel(data.tagged_sents(data_file), HonnibalFeats, CollinsNormalisation, 0.1)

		path = self.working_dir()
		# saved_features = path + '/' + self.get_save_file_name('_features')
		saved_params = path + '/' + self.get_save_file_name('_parameters')

		# Learn features

		self.model.train()

		# Learn feature weights (incrementally... in case we overheat and crash :-/)
		# TODO Use the CV training set to tune the regularization_parameter of the MaxEntMarkovModel i.e. smaller param. learning cycles

		iterations = int(self.config('iterations'))
		for i in range(0, iterations):
			print "Iteration set #", i+1
			not self.model.learn_parameters(maxiter=3)

		# TODO: Use cross-validation set to tune the regularization param.

		return True
