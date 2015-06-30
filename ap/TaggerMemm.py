from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

class TaggerMemm(MachineLearningModule):
	""" Train MEMM tagger.
	"""
	def run(self, _=None):

		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		training = data.tagged_sents(self.config('training_file'))

		print "Training MaxEnt tagger"

		reg = float(self.config('regularization'))
		mxitr = int(self.config('maxiter'))
		num_folds = 10

		subset_size = len(training)/num_folds
		for i in range(num_folds):
			testing = training[i*subset_size:][:subset_size]
			training = training[:i*subset_size] + training[(i+1)*subset_size:]
			self.tagger.train(training, regularization=reg, maxiter=mxitr)

		return True

	def save(self, path = None, filename_prefix = ''):
		return self.tagger.save(self.working_dir(), filename_prefix)

	def load(self, path = None, filename_prefix = ''):
		self.tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		return self.tagger.load(self.working_dir(), filename_prefix)