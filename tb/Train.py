from textblob_aptagger import PerceptronTagger

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

		reader = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		training_file_ids = [self.config(i) for i in ['training_file', 'cross_validation_file']]
		training_data = reader.tagged_sents(training_file_ids)

		sentences = []
		for s in training_data:
			sentences.append(zip(*s))

		# Train the model

		model = PerceptronTagger()
		model.train(sentences=sentences, save_loc=self.working_dir()+'/PerceptronTaggerModel.pickle')

		return True

	def save(self, path):
		# PerceptronTagger does its own saving after training.
		pass

	def load(self, path):
		pass