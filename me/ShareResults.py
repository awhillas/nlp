"""
MaxEnt Markov Model training
"""
from os import path
import cPickle as pickle
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader

class ShareResults(MachineLearningModule):
	""" Load the results of the Shared tagging task to pass on to  the Test module
	"""
	def run(self, previous):
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		original = data.sents(self.get('cv_file'))

		self.load(self.dir('working'))

		assert len(original) == len(self.labeled_sequences.keys())

		return True  # Don't save

	def save(self, data, path = None):
		pass

	def load(self, save_path = None, filename_prefix = ''):
		file_path = save_path + '/memm_tagged_sentences-reg_%.2f.pickle' % self.get('regularization')
		self.labeled_sequences = {}
		if path.exists(file_path):
			self.labeled_sequences.update(pickle.load(open(file_path)))
		else:
			raise IOError('File not found: '+file_path)
