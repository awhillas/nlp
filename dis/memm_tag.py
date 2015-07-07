import dispy
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader

def setup(): # executed on each node before jobs are scheduled
	from lib.ml_framework import Experiment
	from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
	global Experiment, MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation  # stick imports into global scope, create global shared data
	return 0

def compute(model_file, sentence):
	global current_model_file

	if model_file != current_model_file:
		import cPickle as pickle
		tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		tagger.__dict__.update(pickle.load(open(model_file)))
		current_model_file = model_file

	return tagger.tag(sentence)

# # # # # # # # # # # distributed part
class Predict(MachineLearningModule):

	def run(self, previous):
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		model_file = MaxEntMarkovModel.save_file(self.dir('working'), '-reg_%.2f' % float(exp.get('regularization')))

		cluster = dispy.JobCluster(compute, setup=setup, reentrant=True)
		jobs = []
		for sentence in data.sents(self.get('cv_file')):
			job = cluster.submit(self._experiment, model_file, sentence)
			# job.id = i
			jobs.append(job)

		for job in jobs:
			job()
			if job.status != dispy.DispyJob.Finished:
				print('job %s failed: %s' % (job.id, job.exception))
			else:
				print('%s: %s' % (job.id, job.result))