import dispy
import dispy.httpd

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader

def setup(): # executed on each node before jobs are scheduled
	return 0

def train(working_dir, fold_datas, fold_id, reg, mxitr):
	# TODO: Make this work??? Might need this if we want to mess with feature engineering.
	from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
	tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
	tagger.train(fold_datas, regularization=reg, maxiter=mxitr)
	tagger.save(working_dir, '-fold_%02d' % fold_id)


class MemmMultiTrain(MachineLearningModule):

	def run(self, previous):

		# Data

		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		training = data.tagged_sents(self.config('training_file'))
		# testing = data.sents(self.get('cv_file'))

		reg = self.get('regularization')
		mxitr = self.get('maxiter')

		# 10 fold cross validation
		num_folds = 10
		subset_size = len(training)/num_folds

		# Train the model

		cluster = dispy.JobCluster(train, setup=setup, reentrant=True)
		http_server = dispy.httpd.DispyHTTPServer(cluster) # monitor cluster at http://localhost:8181
		jobs = []
		for i in range(num_folds):
			# tagging = training[i*subset_size:][:subset_size]
			learning = training[:i*subset_size] + training[(i+1)*subset_size:]
			job = cluster.submit(self.dir('working'), learning, i, reg, mxitr)
			job.id = i
			jobs.append(job)
		cluster.wait() # wait for all jobs to finish
		http_server.shutdown() # this waits until browser gets all updates
		cluster.close()

		return False  # no saving this module.