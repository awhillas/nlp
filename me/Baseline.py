__author__ = "Alexander Whillas <whillas@gmail.com>"

"""
Baseline tagging. Most frequent tag.
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import CollinsNormalisation
#from lib.pos_tagging import tag_all

class Baseline(MachineLearningModule):
	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.input_module = 'me.Train'
		self.model = None
		self.labeled_sequences = {}

	def run(self, trained):
		self.model = trained.model
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus

		# Baseline model
		self.labeled_sequences = tag_all(
			data.sents(self.config('testing_file')),
			tagger=self.model.frequency_tag,
			normaliser=CollinsNormalisation
		)

		return True