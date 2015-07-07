import dispy, dispy.httpd
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

def setup(): # executed on each node before jobs are scheduled
	from lib.ml_framework import Experiment
	from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
	# stick imports into global scope, create global shared data
	global Experiment, MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation, current_model_file
	current_model_file = None
	return 0

def compute(model_file, sentence):
	global current_model_file

	if model_file != current_model_file:
		import cPickle as pickle
		tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		tagger.__dict__.update(pickle.load(open(model_file)))
		current_model_file = model_file
	return tagger.tag(sentence)


class MemmTag(MachineLearningModule):

	def run(self, previous):
		http_server = None
		labeled_sequences = {}
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		model_file = MaxEntMarkovModel.save_file(self.dir('working'), '-reg_%.2f' % self.get('regularization'))

		cluster = dispy.JobCluster(compute, setup=setup, reentrant=True)
		http_server = dispy.httpd.DispyHTTPServer(cluster) # monitor cluster at http://localhost:8181
		jobs = []
		unlabeled = data.sents(self.get('cv_file'))
		for i, sentence in enumerate(unlabeled):
			job = cluster.submit(model_file, sentence)
			job.id = i
			jobs.append(job)

		if http_server is not None:
			for job in jobs:
				job()
				if job.status != dispy.DispyJob.Finished:
					print('job %s failed: %s' % (job.id, job.exception))
				else:
					print('%s: %s' % (job.id, job.result))
					# print('%s executed job %s at %s with %s\n%s' % (host, job.id, job.start_time, n, job.result))
					labeled_sequences["".join(unlabeled[i])] = job.result
			cluster.stats()
		else:
			cluster.wait() # wait for all jobs to finish
			http_server.shutdown() # this waits until browser gets all updates
			cluster.close()


		self.backup(labeled_sequences, self.dir('working') + '/memm_tagged_sentences-reg_%.2f.pickle' % self.get('regularization'))
		return False