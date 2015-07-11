import dispy, dispy.httpd
import functools
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

def setup(working_dir): # executed on each node ONCE before jobs are scheduled
	global tagger
	from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
	tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
	try:
		tagger.load(working_dir)
	except:
		return 1
	else:
		return 0

def cleanup():
	import gc
	del globals()['tagger']
	gc.collect()

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

		f = functools.partial(setup, self.dir('working'), i)  # make setup function with some parameters
		cluster = dispy.JobCluster(compute, setup=f, cleanup=cleanup, reentrant=True)
		http_server = dispy.httpd.DispyHTTPServer(cluster) # monitor cluster at http://localhost:8181
		jobs = []
		# unlabeled = data.sents(self.get('cv_file'))
		unlabeled = data.sents(self.get('testing_file'))
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
		self.backup(tags, self.dir('working') + '/memm_tagged_testing_sentences-reg_%.2f.pickle' % reg)

		return False

	def load(self, path = None, filename_prefix = ''):
		reg = self.get('regularization')
		self.tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		self.tagger.load(self.dir('working'), '-reg_%.2f' % reg)
		self.labeled_sequences = self.restore(self.dir('working') + '/memm_tagged_sentences-reg_%.2f.pickle' % reg)
		return 0