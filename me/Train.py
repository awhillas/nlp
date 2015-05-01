"""
MaxEnt Markov Model training
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, HonnibalFeats, CollinsNormalisation
import os

class Train(MachineLearningModule):

	def run(self, _):
		# Training
		print "Training MaxEnt model..."
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		data_file = self.config('training_file')  # use smaller set for development
		self.model = MaxEntMarkovModel(data.tagged_sents(data_file), HonnibalFeats, CollinsNormalisation, 0.1)

		path = self.working_dir()
		saved_features = path + '/' + self.get_save_file_name('_features')
		saved_params = path + '/' + self.get_save_file_name('_parameters')

		# Learn features

		if not os.path.isfile(saved_features):
			self.model.train()
			self.save(filename_prefix='_features')
		else:
			self.load(filename_prefix='_features')

		# Learn feature weights (incrementally... in case it crashes)

		if not os.path.isfile(saved_params):
			for i in range(0, 1):
				print "Iteration set #", i
				self.model.learn_parameters(maxiter=10)
				self.save(filename_prefix='_parameters')
		else:
			self.load(filename_prefix='_parameters')

		# TODO: Use cross-validation set to tune the regularization param.

		return True
