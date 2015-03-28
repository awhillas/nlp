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
		# Use both cross validation and training sets since we are not tuning model params.
		training_file_ids = [self.config(i) for i in ['training_file', 'cross_validation_file']]
		for s in reader.tagged_sents(training_file_ids):
			sentences.append(zip(*s))

		# Train the model

		pt = PerceptronTagger()
		pt.train(sentences=sentences, save_loc=self.working_dir()+'/PerceptronTaggerModel.pickle')

		return True

	def save(self, path):
		# PerceptronTagger does its own saving after training.
		pass

	def load(self, path):
		pt = PerceptronTagger()
		pt.load(self.pickle_file())
		return pt