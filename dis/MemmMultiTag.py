import dispy
import dispy.httpd
import functools

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader

def setup(working_dir, fold_id): # executed on each node before jobs are scheduled
	from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
	global tagger
	tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
	tagger.load(working_dir, '-fold_%02d' % fold_id)
	return 0

def multi_tag(sentence):
	s, _ = map(list, zip(*sentence))
	# single = tagger.tag(s)
	multi = tagger.multi_tag(s)  # get all multi tags and then filter them later to tune ambiguity.
	# return single, multi
	return multi

class MemmMultiTag(MachineLearningModule):

	def run(self, _):
		self.tagged = []

		# Data
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		training = data.tagged_sents(self.get('training_file'))
		# testing = data.sents(self.get('cv_file'))

		num_folds = 10  # 10 fold cross validation
		subset_size = len(training)/num_folds
		http_server = None

		# Tag the 1-left-out folds and accumulate for parser training.


		for i in range(num_folds):
			tagging = training[i*subset_size:][:subset_size]
			# learning = training[:i*subset_size] + training[(i+1)*subset_size:]
			f = functools.partial(setup, self.dir('working'), i)  # make setup function with some parameters
			cluster = dispy.JobCluster(multi_tag, setup=f, reentrant=True)

			# Monitor cluster

			if http_server is None:
				http_server = dispy.httpd.DispyHTTPServer(cluster) # monitor cluster(s) at http://localhost:8181
			else:
				http_server.add_cluster(cluster)

			# Create Jobs

			jobs = []
			for j, sentence in enumerate(tagging):
				job = cluster.submit(sentence)
				job.id = (i+1) * (j+1)
				jobs.append(job)
			cluster.wait() # wait for all jobs to finish
			http_server.del_cluster(cluster)  # else we get an error when we try to re-add it
			cluster.close()

			# Collect the results in a single set.

			for job in jobs:
				job()
				if job.status != dispy.DispyJob.Finished:
					raise Exception('job %s failed: %s' % (job.id, job.exception))
				else:
					multi = job.result
					self.tagged.append(multi)
			self.save(self.tagged)
		http_server.shutdown()

		return True  # call .save() when done.

	def save(self, data, path = None):
		self.backup(data, self._backup_file_path())

	def load(self, path = None, filename_prefix = ''):
		# self.tagged = self.restore(self._backup_file_path())
		pass

	def _backup_file_path(self):
		return self.dir('working') + '/memm_multi-tagged_sentences-reg_%.2f.pickle' % self.get('regularization')