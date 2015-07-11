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
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		training_data = data.tagged_sents(self.get('training_file'))
		# cv_data = data.tagged_sents(self.get('cross_validation_file')) # TODO: Use cross-validation set to tune the regularization param.
		reg = self.get('regularization')
		mxitr = self.get('maxiter')

		print "Training MaxEnt model..."

		# Learn features
		self.tagger.train(training_data, regularization=reg, maxiter=mxitr)
		self.log("MaxEnt Features", "{:,}".format(len(self.tagger.weights)))

		return True

	def save(self, path = None, filename_prefix=None):
		return self.tagger.save(self.working_dir(), filename_prefix='-reg_%.2f' % float(self.get('regularization')))

	def load(self, path = None, filename_prefix=None):
		self.tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		try:
			self.tagger.load(self.working_dir(), filename_prefix='-reg_%.2f' % float(self.get('regularization')))
		except:
			return None
		else:
			return True