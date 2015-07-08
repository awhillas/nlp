import dispy
import dispy.httpd

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

def setup(): # executed on each node before jobs are scheduled
	from lib.ml_framework import Experiment
	from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
	# stick imports into global scope, create global shared data
	global Experiment, MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation, current_fold
	current_fold = None
	return 0

def multi_tag(working_dir, fold_id, sentence):
	global current_fold
	tagger = None

	if fold_id != current_fold:
		tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		tagger.load(working_dir, '-fold_%02d' % fold_id)
		current_fold = fold_id

	s, _ = map(list, zip(*sentence))
	single = tagger.tag(s)
	multi = tagger.multi_tag(s)  # get all multi tags and then filter them later to tune ambiguity.
	return (single, multi)


class MemmMultiTag(MachineLearningModule):

	def run(self, previous):
		# Data
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		training = data.tagged_sents(self.config('training_file'))
		# testing = data.sents(self.get('cv_file'))

		reg = self.get('regularization')
		mxitr = self.get('maxiter')
		ambiguity = self.get('ambiguity')

		# 10 fold cross validation
		num_folds = 10
		subset_size = len(training)/num_folds
		training_tags = []  # Output: master set of CV tags built from the leave-1-out sets

		# Tag the 1-left-out folds

		cluster = dispy.JobCluster(multi_tag, setup=setup, reentrant=True)
		http_server = dispy.httpd.DispyHTTPServer(cluster) # monitor cluster at http://localhost:8181
		jobs = []
		for i in range(num_folds):
			tagging = training[i*subset_size:][:subset_size]
			# learning = training[:i*subset_size] + training[(i+1)*subset_size:]
			for i, sentence in enumerate(tagging):
				job = cluster.submit(self.dir('working'), i, sentence)
				job.id = i
				jobs.append(job)
			cluster.wait() # wait for all jobs to finish
			http_server.shutdown() # this waits until browser gets all updates
			cluster.close()
			# collect the results in a single set.
			for job in jobs:
				job()
				if job.status != dispy.DispyJob.Finished:
					print('job %s failed: %s' % (job.id, job.exception))
				else:
					print('%s: %s' % (job.id, job.result))
					training_tags.append(job.result)

		self.backup(training_tags, self.dir('working') + '/memm_tagged_sentences-reg_%.2f.pickle' % self.get('regularization'))
		return False