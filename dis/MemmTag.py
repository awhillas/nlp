import dispy, dispy.httpd
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

def setup(): # executed on each node before jobs are scheduled
	from lib.ml_framework import Experiment
	from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
	# stick imports into global scope, create global shared data
	global Experiment, MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation, current_reg
	current_reg = None
	return 0

def compute(working_dir, reg, sentence):
	global current_reg

	if reg != current_reg:
		tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		tagger.load(working_dir, '-reg_%.2f' % reg)
		current_reg = reg
	return tagger.tag(sentence)


class MemmTag(MachineLearningModule):

	def run(self, previous):

		def save_data(jobs):
			labeled_sequences = [[]] * len(jobs)
			for job in jobs:
				job()
				if job.status != dispy.DispyJob.Finished:
					print('job %s failed: %s' % (job.id, job.exception))
				else:
					print('%s: %s' % (job.id, job.result))
					# print('%s executed job %s at %s with %s\n%s' % (host, job.id, job.start_time, n, job.result))
					labeled_sequences[job.id] = job.result
			return labeled_sequences


		http_server = None
		reg = self.get('regularization')
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		cluster = dispy.JobCluster(compute, setup=setup, reentrant=True)
		http_server = dispy.httpd.DispyHTTPServer(cluster) # monitor cluster at http://localhost:8181
		jobs = []
		unlabeled = data.sents(self.get('cv_file'))
		for i, sentence in enumerate(unlabeled):
			job = cluster.submit(self.dir('working'), reg, sentence)
			job.id = i
			jobs.append(job)

		if http_server is not None:
			cluster.wait() # wait for all jobs to finish
			cluster.stats()
			http_server.shutdown() # this waits until browser gets all updates
			cluster.close()

		tags = save_data(jobs)
		self.backup(tags, self.dir('working') + '/memm_tagged_sentences-reg_%.2f.pickle' % reg)

		return False

	def load(self, path = None, filename_prefix = ''):
		reg = self.get('regularization')
		self.tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		self.tagger.load(self.dir('working'), '-reg_%.2f' % reg)
		self.labeled_sequences = self.restore(self.dir('working') + '/memm_tagged_sentences-reg_%.2f.pickle' % reg)
		return 0