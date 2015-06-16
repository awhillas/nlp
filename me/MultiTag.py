__author__ = "Alexander Whillas <whillas@gmail.com>"

"""
MultiTagging which feeds into the Parser.
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
import pandas

class MultiTag(MachineLearningModule):
	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.input_module = 'me.Train'
		self.model = None
		self.labeled_sequences = {}

	def run(self, trained):
		self.model = trained.model
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus

		# Baseline model
		for s in data.sents(self.config('testing_file')):
			print s
			fwd, bkw, posterior = self.model.tag_probability_distributions(s)
		# self.labeled_sequences = self.model.for_all(
		# 	data.sents(self.config('testing_file')),
		# 	self.model.tag_probability_distributions
		# )

		return True