from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

def train(working_dir, fold_datas, fold_id, reg, mxitr):
	tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
	tagger.train(fold_datas, regularization=reg, maxiter=mxitr)
	tagger.save(working_dir, '-fold_%02d' % fold_id)


class MaxEntFoldTrain(MachineLearningModule):

	def run(self, previous):
		# Data
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		training = data.tagged_sents(self.config('training_file'))
		reg = self.get('regularization')
		mxitr = self.get('maxiter')

		# 10 fold cross validation
		num_folds = 10
		subset_size = len(training)/num_folds
		i = self.get('fold')
		# tagging = training[i*subset_size:][:subset_size]
		learning = training[:i*subset_size] + training[(i+1)*subset_size:]
		train(self.dir('working'), learning, i, reg, mxitr)

		return False  # no saving this module.