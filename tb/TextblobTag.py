""" POS tagger using the TextBlob library
	requires:
		pip install -U textblob
		pip install -U textblob-aptagger
	Using predominantly for bench marking
"""

from __future__ import print_function

from textblob import TextBlob
from textblob_aptagger import PerceptronTagger

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader


class TextblobTag(MachineLearningModule):
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'tb.TextblobTrain'

	def run(self, tagger):

		# Init

		reader = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		tagger = PerceptronTagger(load=False)
		tagger.load(loc=self.get_pickle_file())

		# generate tags
		with open(self.config('output_file'), 'w') as f:
			for s in reader.tagged_sents(self.config('testing_file')):
				blob = TextBlob(" ".join(s.words()), pos_tagger=tagger)
				words, tags = zip(*blob.tags)
				print(" ".join(tags)+"\n", f)

		return True