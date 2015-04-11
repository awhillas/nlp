"""
MaxEnt Markov Model training
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, HonibbalsFeats, CollinsNormalisation
import os

class Train(MachineLearningModule):

	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.model = MaxEntMarkovModel(HonibbalsFeats, CollinsNormalisation, 0.5)

	def run(self, _):
		# Training
		print "Training MEMM..."

		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		#data_file = self.config('training_file')
		data_file = self.config('cross_validation_file')  # use smaller set for development
		saved_features = self.working_dir() + '/' + self.get_save_file_name('_features')

		# Learn features

		if not os.path.isfile(saved_features):
			self.model.train(data.tagged_sents(data_file))
			self.save(filename_prefix='_features')

		# Learn feature weights

		if os.path.isfile(saved_features):
			self.load(filename_prefix='_features')
		self.model.learn_parameters(data.tagged_sents(data_file))

		# Use cross-validation set to tune the regularization param.

		return True