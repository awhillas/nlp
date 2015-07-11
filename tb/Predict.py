""" POS tagger using the TextBlob library
	requires:
		pip install -U textblob
		pip install -U textblob-aptagger
	Using predominantly for bench marking
"""

from lib.PerceptronTagger import PerceptronTagger
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.pos_tagging import matrix_to_string


class Predict(MachineLearningModule):
	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.input_module = 'tb.Train'
		self.labeled_sequences = []

	def run(self, _):

		# Init

		reader = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		tron = PerceptronTagger(load=False)
		tron.load(loc=self.working_dir()+'/PerceptronTaggerModel.pickle')

		# generate tags

		for s in reader.sents(self.get('testing_file')):
			predicted = tron.tag(s)
			if len(predicted) > 0:
				words, tags = zip(*predicted)
				print matrix_to_string([words, tags]), "\n"
				self.labeled_sequences += [predicted]
			else:
				print "Could not tag sentence?", " ".join(words)

		return True