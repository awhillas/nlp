from lib.PerceptronTagger import PerceptronTagger
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader


class Train(MachineLearningModule):
	""" Train Textblob Perceptron on our own data so the results are comparable.
	"""
	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.model = None

	def run(self, _=None):

		# Get (words, tags) sequences for all sentences

		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		training_data = data.tagged_sents(self.config('training_file'))

		sentences = []
		for s in training_data:
			sentences.append(zip(*s))

		# Train the model

		self.model = PerceptronTagger(load=False)
		self.model.train(sentences=sentences, save_loc=self.working_dir()+'/PerceptronTaggerModel.pickle')

		return True

	def save(self, path=None):
		# PerceptronTagger does its own saving after training.
		pass

	def load(self, path=None):
		pass  # tb.Predict handles this