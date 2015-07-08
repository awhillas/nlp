import dispy
import dispy.httpd
import functools.partial

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

def setup(working_dir, fold_id): # executed on each node before jobs are scheduled
	from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
	global tagger
	tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
	tagger.load(working_dir, '-fold_%02d' % fold_id)
	return 0

def multi_tag(sentence):
	s, _ = map(list, zip(*sentence))
	single = tagger.tag(s)
	multi = tagger.multi_tag(s)  # get all multi tags and then filter them later to tune ambiguity.
	return single, multi

class MemmMultiTag(MachineLearningModule):

	def run(self, previous):
		# Data
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		training = data.tagged_sents(self.config('training_file'))
		# testing = data.sents(self.get('cv_file'))


		num_folds = 10  # 10 fold cross validation
		subset_size = len(training)/num_folds
		training_tags = []  # Output: master set of CV tags built from the leave-1-out sets
		training_tags_multi = []

		# Tag the 1-left-out folds

		for i in range(num_folds):
			tagging = training[i*subset_size:][:subset_size]
			# learning = training[:i*subset_size] + training[(i+1)*subset_size:]
			f = functools.partial(setup, self.dir('working'), i)
			cluster = dispy.JobCluster(multi_tag, setup=f, reentrant=True)
			http_server = dispy.httpd.DispyHTTPServer(cluster) # monitor cluster at http://localhost:8181
			jobs = []
			for j, sentence in enumerate(tagging):
				job = cluster.submit(self.dir('working'), j, sentence)
				job.id = i * j
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
					single, multi = job.result
					training_tags.append(single)
					training_tags_multi.append(multi)

		self.backup(training_tags, self.dir('working') + '/memm_tagged_sentences-reg_%.2f.pickle' % self.get('regularization'))
		self.backup(training_tags_multi, self.dir('working') + '/memm_multi-tagged_sentences-reg_%.2f.pickle' % self.get('regularization'))
		return False