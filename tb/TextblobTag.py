""" POS tagger using the TextBlob library
	requires:
		pip install -U textblob
		pip install -U textblob-aptagger
	Using predominantly for bench marking
"""

from __future__ import print_function
from ml_framework import MachineLearningModule
from textblob import TextBlob
from textblob_aptagger import PerceptronTagger
from conllu import ConlluReader

class TextblobTag(MachineLearningModule):
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'tb.TextblobTrain'

	def run(self, tagger):

		# Init

		reader = ConlluReader(self.config.get(self.data_id, 'uni_dep_base'), '.*\.conllu')  # Corpus
		tagger = PerceptronTagger(load=False)
		tagger.load(self.get_output_file_name())

		# generate tags
		with open(self.get_output_file_name(), "w") as f:
			for s in reader.tagged_sents(self.config.get(self.data_id, 'testing_file')):
				blob = TextBlob(" ".join(s.words()), pos_tagger=tagger)
				words, tags = zip(*blob.tags)
				print(" ".join(tags), f)

		return True