import dispy, dispy.httpd
import functools
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader

def setup(working_dir, reg): # executed on each node ONCE before jobs are scheduled
	from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
	global tagger
	tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
	tagger.load(working_dir, '-reg_%.2f' % reg, True)
	return 0

def cleanup():
	import gc
	del globals()['tagger']
	gc.collect()

def tag(sentence):
	return tagger.tag(sentence)

def multi_tag(sentence):
	multi = tagger.multi_tag(sentence, 0.01)  # remove the very very unlikely
	return {"".join(sentence): multi}

class MemmTag(MachineLearningModule):

	def run(self, previous):

		def save_jobs_list_data(jobs):
			labeled_sequences = [[]] * len(jobs)
			for job in jobs:
				job()
				if job.status != dispy.DispyJob.Finished:
					print('job %s failed: %s' % (job.id, job.exception))
				else:
					# print('%s: %s' % (job.id, job.result))
					labeled_sequences[job.id] = job.result
			return labeled_sequences

		def save_jobs_dict_data(jobs):
			data = {}
			for job in jobs:
				job()
				if job.status != dispy.DispyJob.Finished:
					print('job %s failed: %s' % (job.id, job.exception))
				else:
					# print('%s: %s' % (job.id, job.result))
					data.update(job.result)
			return data


		reg = self.get('regularization')
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus

		http_server = None
		for data_name in ['testing_file', 'cv_file']:

			for tagging_type in [tag, multi_tag]:

				unlabeled = data.sents(self.get(data_name))

				func = functools.partial(setup, self.dir('working'), reg)  # make setup function with some parameters
				cluster = dispy.JobCluster(tagging_type, setup=func, cleanup=cleanup, reentrant=True)

				if http_server is None:
					http_server = dispy.httpd.DispyHTTPServer(cluster) # monitor cluster(s) at http://localhost:8181
				else:
					http_server.add_cluster(cluster)

				jobs = []
				for i, sentence in enumerate(unlabeled):
					job = cluster.submit(sentence)
					job.id = i
					jobs.append(job)

				if http_server is not None:
					cluster.wait() # wait for all jobs to finish
					cluster.stats()
					http_server.shutdown() # this waits until browser gets all updates
					cluster.close()

				if tagging_type == tag:
					tags = save_jobs_list_data(jobs)
					data_type = '1-best'
				else:
					tags = save_jobs_dict_data(jobs)
					data_type = 'multi'
				self.backup(tags, tagged_file_name(self.dir('working'), data_name, data_type, reg))

		if http_server:
			http_server.shutdown()
		return True

	def load(self, path = None, filename_prefix = ''):
		pass

	def save(self, data, path = None):
		pass

def tagged_file_name(working_dir, data_name, data_type, reg):
	return working_dir + '/tagged_sentences_%s_%s-reg_%.2f.pickle' % (data_name, data_type, reg)