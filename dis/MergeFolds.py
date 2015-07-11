from lib.ml_framework import MachineLearningModule
from os import path
from dis.MemmMultiTag import backup_file_path
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

class MergeFolds(MachineLearningModule):
	""" Merge the results of the 10-fold multi-tagging
	"""
	def run(self, previous):
		num_folds = 10  # 10 fold cross validation
		self.mult_tagged = []
		reg = self.get('regularization')
		for i in range(num_folds):
			file_path = backup_file_path(self.dir('working'), i, reg)
			if path.exists(file_path + ".gz"):
				self.mult_tagged += self.restore(file_path)
			else:
				raise Exception("Fold %d data missing!" % i)

		tagger = load_memm_tagger(self.dir('working'), self.get('regularization'))
		self.all_tags = tagger.get_classes()

		return True

def load_memm_tagger(working_dir, reg=0.66):
	tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
	tagger.load(working_dir, '-reg_%.2f' % reg)
	return tagger