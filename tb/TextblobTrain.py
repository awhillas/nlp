
from ml_framework import MachineLearningModule
from textblob import TextBlob
from textblob_aptagger import PerceptronTagger
from nltk.tree import Tree

class TextblobTrain(MachineLearningModule):
	""" Train Textblob Perceptron on our own data so the results are comparable.
	"""
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)

	def run(self, _=None):
		# Get the sentences and their tags into an agreeable shape
		sentences = []
		with open(self.get_input_file_name(), 'r') as f:
			for line in f:
				tree = Tree.fromstring(line)
				sentences.append(zip(*tree.pos()))
		pt = PerceptronTagger()
		pt.train(sentences=sentences, save_loc=self.get_output_file_name())

		return True

	def save(self, path):
		# PerceptronTagger does its own saving after training.
		pass

	def load(self, path):
		pt = PerceptronTagger()
		pt.load(self.get_output_file_name())
		return pt