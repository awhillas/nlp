from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import PerceptronParser
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

class Train(MachineLearningModule):
	""" Train Textblob Perceptron on our own data so the results are comparable.
	"""
	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.model = None

	def run(self, _=None):

		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus

		# Use the MaxEnt tagger
		tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		tagger.train(data=data.tagged_sents(self.config('training_file')))

		# Train the model
		self.model = PerceptronParser(tagger=tagger)
		self.model.train(data.parsed_sents(self.config('training_file')))

		return True