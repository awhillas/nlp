__author__ = "Alexander Whillas <whillas@gmail.com>"

"""
Baseline tagging. Most frequent tag.
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader

class Baseline(MachineLearningModule):
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'me.Train'
		self.model = None
		self.labeled_sequences = {}

	def run(self, trained):
		self.model = trained.model
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus

		# Baseline model
		self.labeled_sequences = self.model.for_all(
			data.sents(self.config('testing_file')),
			self.model.frequency_tag
		)

		return True