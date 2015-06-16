import random
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

class MEMMTagger(MachineLearningModule):
	""" Train Textblob Perceptron on our own data so the results are comparable.
	"""
	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)

	def run(self, _=None):

		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		tagged_sentences = data.tagged_sents(self.config('training_file'))
		reg = float(self.config('regularization'))
		mxitr = int(self.config('maxiter'))

		print "MaxEnt tagger"
		if not self.tagger.load(self.working_dir()):
			print "...training"
			self.tagger.train(tagged_sentences, regularization=reg, maxiter=mxitr)

		if not self._experiment.no_cache:
			self.tagger.save(self.working_dir())

		return True
