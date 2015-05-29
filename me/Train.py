__author__ = "Alexander Whillas <whillas@gmail.com>"

"""
MaxEnt Markov Model (MEMM) training
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
import time

class Train(MachineLearningModule):
	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.model = MaxEntMarkovModel(Ratnaparkhi96Features, CollinsNormalisation)

	def run(self, _):
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		training_data = data.tagged_sents(self.config('training_file'))

		if self._experiment.no_cache or not self.load(filename_prefix="_memm"):
			print "Training MaxEnt model..."
			# Get data

			# Learn features
			self.model.train(training_data)
			self._experiment.output["Features"] = len(self.model.weights)
			print "Features:", len(self.model.weights)

		# Learn feature weights

		# TODO Use the CV training set to tune the regularization_parameter of the MaxEntMarkovModel i.e. smaller param. learning cycles

		iterations = int(self.config('iterations'))
		reg = float(self.config('regularization'))
		mxitr = int(self.config('maxiter'))

		for i in range(0, iterations):
			print "Iteration set #", i+1, "of", iterations  # incrementally... in case we overheat and crash :-/
			self.model.learn_parameters(training_data, regularization=reg, maxiter=mxitr)
			if not self._experiment.no_save:
				self.save(filename_prefix="_memm")
			time.sleep(5)

		self.delete(filename_prefix="_memm")  # Remove temp backup file

		# TODO: Use cross-validation set to tune the regularization param.
		# cv_data = data.tagged_sents(self.config('cross_validation_file'))

		return self.model
