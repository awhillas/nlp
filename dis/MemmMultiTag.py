import dispy
import dispy.httpd
import functools
import cPickle as pickle
import gzip
import os
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader

def setup(working_dir, fold_id): # executed on each node ONCE before jobs are scheduled
	global tagger
	from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
	tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
	try:
		tagger.load(working_dir, '-fold_%02d' % fold_id)
	except:
		return 1
	else:
		return 0

def cleanup():
	import gc
	del globals()['tagger']
	gc.collect()

def multi_tag(sentence):
	s, _ = map(list, zip(*sentence))
	# single = tagger.tag(s)
	multi = tagger.multi_tag(s, 0.01)  # remove the very unlikely
	# return single, multi
	return {"".join(sentence): multi}

def decompress_model(self, fold_id):
	file_name = MaxEntMarkovModel.save_file(self.dir('working'), '-fold_%02d' % fold_id)
	# file_name_gz = file_name + ".gz"
	if os.path.exists(file_name):
		print "Decompressing %s" % file_name
		with gzip.open(file_name) as f_in:
			data = pickle.load(f_in)
			with open(file_name, 'wb') as f_out:
				pickle.dump(data, f_out, -1)
				print "Decompressed to %s" % file_name
				return file_name
	else:
		raise Exception("File don't exist, yo? %s" % file_name)

class MemmMultiTag(MachineLearningModule):

	def run(self, _):
		self.tagged = {}

		# Data
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		training = data.tagged_sents(self.get('training_file'))
		# testing = data.sents(self.get('cv_file'))

		num_folds = 10  # 10 fold cross validation
		subset_size = len(training) / num_folds
		http_server = None

		# Tag the 1-left-out folds and accumulate for parser training.

		for i in range(num_folds):
			print "Fold", i
			skip_fold = True if os.path.exists(backup_file_path(self.dir('working'), i, self.get('regularization'))+".gz", ) else False  # save time is we've done it already
			if not skip_fold:
				current_model_file = decompress_model(self, 0)  # unzip model
				tagging = training[i*subset_size:][:subset_size]
				# learning = training[:i*subset_size] + training[(i+1)*subset_size:]
				f = functools.partial(setup, self.dir('working'), i)  # make setup function with some parameters
				cluster = dispy.JobCluster(multi_tag, setup=f, cleanup=cleanup, reentrant=True)

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
				# next_model_file = decompress_model(self, i+1)  # unzip next model while waiting
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
						self.tagged.update(multi)
				self.save(self.tagged, i)
				os.remove(current_model_file)  # remove unzipped version
				# current_model_file = next_model_file
			else:
				continue
		http_server.shutdown()
		return False

	def save(self, data, fold_id = 0):
		self.backup(data, backup_file_path(self.dir('working'), fold_id, self.get('regularization')))

	def load(self, path = None, filename_prefix = ''):
		# self.tagged = self.restore(backup_file_path(0, self.get('regularization')))
		pass

def backup_file_path(working_dir, fold_id, reg):
	return working_dir + '/memm_multi-tagged_sentences-reg_%.2f-fold_%d.pickle' % (reg, fold_id)
