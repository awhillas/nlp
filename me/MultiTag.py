__author__ = "Alexander Whillas <whillas@gmail.com>"

"""
MultiTagging which feeds into the Parser.
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

class MultiTag(MachineLearningModule):
	PREVIOUS_MODULE = 'me.Train'

	def run(self, previous):
		self.model = previous.model
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus

		for s in data.sents(self.config('testing_file')):
			self.labeled_sequences = self.model.multi_tag(s)

		return True

	def save(self, path = None, filename_prefix = ''):
		return self.model.save(self.working_dir(), filename_prefix)

	def load(self, path = None, filename_prefix = ''):
		self.model = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		return self.model.load(self.working_dir(), filename_prefix)