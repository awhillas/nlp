__author__ = "Alexander Whillas <whillas@gmail.com>"

"""
MaxEnt Markov Model (MEMM) training
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
import time

class Train(MachineLearningModule):
	def run(self, _):
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		training_data = data.tagged_sents(self.config('training_file'))
		# cv_data = data.tagged_sents(self.config('cross_validation_file')) # TODO: Use cross-validation set to tune the regularization param.
		reg = self.log_me('Reg.', float(self.config('regularization')))
		mxitr = self.log_me('Max iter.', int(self.config('maxiter')))

		print "Training MaxEnt model..."

		# Learn features
		self.model.train(training_data, regularization=reg, maxiter=mxitr)
		self.log_me("Features", len(self.model.weights))  # TODO: doesn't seem to make it ?

		return True

	def load(self, path = None, filename_prefix = ''):
		self.model = MaxEntMarkovModel(Ratnaparkhi96Features, CollinsNormalisation)
		self.model.load(path, filename_prefix)

	def save(self, path = None, filename_prefix = ''):
		self.model.save(path, filename_prefix)