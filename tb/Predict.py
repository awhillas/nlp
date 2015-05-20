""" POS tagger using the TextBlob library
	requires:
		pip install -U textblob
		pip install -U textblob-aptagger
	Using predominantly for bench marking
"""

#from __future__ import print_function

from textblob import TextBlob
from textblob_aptagger import PerceptronTagger

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader


class Predict(MachineLearningModule):
	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.input_module = 'tb.TextblobTrain'

	def run(self, _):

		# Init

		reader = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		tagger = PerceptronTagger(load=False)
		tagger.load(loc=self.working_dir()+'/PerceptronTaggerModel.pickle')

		# generate tags

		for s in reader.tagged_sents(self.config('testing_file')):
			words, gold_tags = zip(*s)
			print " ".join(gold_tags)

			# Tag the sentence and save it.

			blob = TextBlob(" ".join(words), pos_tagger=tagger)
			if len(blob.tags) > 0:
				self.labeled_sequences = blob.tags
			else:
				print "Could not tag sentence?", " ".join(words)

		return True