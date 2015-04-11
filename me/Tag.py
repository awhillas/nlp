"""
MaxEnt Markov Model training
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, HonibbalsFeats

class Tag(MachineLearningModule):
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'me.Train'
		self.labled_sequences = {}

	def run(self, trained):
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		model = trained.model
		self.labled_sequences = model.label(data.sents(self.config('testing_file')))
