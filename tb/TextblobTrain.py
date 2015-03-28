from textblob_aptagger import PerceptronTagger

from lib.ml_framework import MachineLearningModule

from lib.conllu import ConlluReader


class TextblobTrain(MachineLearningModule):
	""" Train Textblob Perceptron on our own data so the results are comparable.
	"""
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)

	def run(self, _=None):

		# Get (words, tags) sequences for all sentences

		reader = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		sentences = []
		training_file_ids = [self.config('training_file'), self.config('cross_validation_file')]
		for s in reader.tagged_sents(training_file_ids):
			sentences.append(zip(*s))

		# Train the model

		pt = PerceptronTagger()
		pt.train(sentences=sentences, save_loc=self.get_pickle_file())

		return True

	def save(self, path):
		# PerceptronTagger does its own saving after training.
		pass

	def load(self, path):
		pt = PerceptronTagger()
		pt.load(self.get_pickle_file())
		return pt