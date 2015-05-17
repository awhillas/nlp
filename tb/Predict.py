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


class Predict(MachineLearningModule):
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'tb.TextblobTrain'

	def run(self, tagger):

		# Init

		reader = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		tagger = PerceptronTagger(load=False)
		tagger.load(loc=self.working_dir()+'/PerceptronTaggerModel.pickle')

		# generate tags
		# Using the MaxEnt model
		self.labeled_sequences = self.model.for_all(
			data.sents(self.config('testing_file')),
			self.model.label
		)

		with open(self.working_dir()+'/'+self.config('output_file'), 'w') as f1:
			with open(self.working_dir()+'/'+self.config('gold_output'), 'w') as f2:
				for s in reader.tagged_sents(self.config('testing_file')):
					words, gold_tags = zip(*s)
					print(" ".join(gold_tags), file=f2)

					# Tag the sentence and save it.

					blob = TextBlob(" ".join(words), pos_tagger=tagger)
					if len(blob.tags) > 0:
						_, tags = zip(*blob.tags)
						print(" ".join(tags), file=f1)
					else:
						print("", file=f1)

		return True