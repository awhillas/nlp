"""
MaxEnt Markov Model training
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.pos_tagging import tag_all

class Predict(MachineLearningModule):
	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.input_module = 'me.Train'
		self.labeled_sequences = []

	def run(self, trained):
		self.model = trained.model
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus

		# Using the MaxEnt model
		self.labeled_sequences = tag_all(
			data.sents(self.config('testing_file')),
			tagger=self.model.label,
			normaliser=self.model.normaliser
		)
		return True