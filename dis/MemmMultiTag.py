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
	from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
	global tagger
	tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
	tagger.load(working_dir, '-fold_%02d' % fold_id)
	return 0

def cleanup():
	import gc
	del globals()['tagger']
	gc.collect()

def multi_tag(sentence):
	s, _ = map(list, zip(*sentence))
	multi = tagger.multi_tag(s, 0.01)  # remove the very very unlikely
	return {"".join(s): multi}

def decompress_model(self, fold_id):
	file_name = MaxEntMarkovModel.save_file(self.dir('working'), '-fold_%02d' % fold_id)
	file_name_gz = file_name + ".gz"
	if not os.path.exists(file_name):
		if os.path.exists(file_name_gz):
			print "Decompressing %s" % file_name_gz
			with gzip.open(file_name_gz) as f_in:
				data = pickle.load(f_in)
				with open(file_name, 'wb') as f_out:
					pickle.dump(data, f_out, -1)
					print "Decompressed to %s" % file_name
					return file_name
		else:
			raise Exception("File don't exist, yo? %s" % file_name_gz)
	else:
		print "File already decompressed?", file_name
		return file_name

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
		reg = self.get('regularization')

		# Tag the 1-left-out folds and accumulate for parser training.

		for i in range(num_folds):
			skip_fold = True if os.path.exists(backup_file_path(self.dir('working'), i, reg)+".gz", ) else False  # save time is we've done it already
			if not skip_fold:
				current_model_file = decompress_model(self, i)  # unzip model
				print "Fold", i, "from", i*subset_size, "to", i*subset_size+subset_size
				unsorted_tagging = training[i*subset_size:][:subset_size]
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
				reversed_n_kulled = list(reversed(sorted(unsorted_tagging, key=lambda k: len(k))))[3:]  # reverse sort by length & kull the top 3 longest
				for j, sentence in enumerate(reversed_n_kulled):  # sort longest first?
					job = cluster.submit(sentence)
					job.id = (i+1) * (j+1)
					jobs.append(job)
				# next_model_file = decompress_model(self, i+1)  # unzip next model while waiting
				cluster.wait() # wait for all jobs to finish
				http_server.del_cluster(cluster)  # else we get an error when we try to re-add it
				cluster.close()

				# Collect the results in a single set.
				failed = 0
				print len(jobs), "jobs"
				for job in jobs:
					job()
					if job.status != dispy.DispyJob.Finished:
						failed += 1
						# raise Exception('job %s failed: %s' % (job.id, job.exception))
					else:
						self.tagged.update(job.result)
				print failed, "failed jobs"
				self.save(self.tagged, i)
				os.remove(current_model_file)  # remove unzipped version
				# current_model_file = next_model_file
			else:
				continue
		if http_server:
			http_server.shutdown()
		return True

	def save(self, data, fold_id = 0):
		pass

	def load(self, file_path = None, filename_prefix = ''):
		# Merge folds into one.
		num_folds = 10  # 10 fold cross validation
		self.mult_tagged = {}
		reg = self.get('regularization')
		for i in range(num_folds):
			file_path = backup_file_path(self.dir('working'), i, reg)
			if os.path.exists(file_path + ".gz"):
				self.mult_tagged.update(self.restore(file_path))
			else:
				raise Exception("Fold %d data missing!: %s" % (i, file_path + ".gz"))

		tagger = load_memm_tagger(self.dir('working'), self.get('regularization'))
		self.all_tags = tagger.get_classes()

def load_memm_tagger(working_dir, reg=0.66):
	tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
	tagger.load(working_dir, '-reg_%.2f' % reg, True)
	return tagger

def backup_file_path(working_dir, fold_id, reg):
	return working_dir + '/memm_multi-tagged_sentences-reg_%.2f-fold_%d.pickle' % (reg, fold_id)
