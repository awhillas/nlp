
from ml_framework import MachineLearningModule
from textblob_aptagger import PerceptronTagger
from conllu import ConlluReader

class TextblobTrain(MachineLearningModule):
	""" Train Textblob Perceptron on our own data so the results are comparable.
	"""
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)

	def run(self, _=None):
		# Get the sentences and their tags into an agreeable shape
		reader = ConlluReader(self.config.get(self.data_id, 'uni_dep_base'), '.*\.conllu')
		sentences = []
		for s in reader.tagged_sents('en-ud-train.conllu'):
			sentences.append(zip(*s))
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