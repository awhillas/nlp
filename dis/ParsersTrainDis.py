import os
import dispy
import functools
from lib.ml_framework import MachineLearningModule
from dis.MemmMultiTag import backup_file_path


def setup(data_base_place, training_file, workin_dir, reg):
	from lib.conllu import ConlluReader
	from lib.PerceptronParser import AmbiguousParser
	from lib.ml_framework import MachineLearningModule
	from ap import dep_tree_to_list

	global parsed_sentences, multi_tags_sentences, parser, working_dir

	working_dir = workin_dir
	parser = AmbiguousParser()
	data = ConlluReader(data_base_place, '.*\.conllu')  # Corpus
	parsed_sentences = [(dep_tree_to_list(dep_tree)) for dep_tree in data.parsed_sents(training_file)]
	multi_tags_sentences = MachineLearningModule.restore(working_dir+'/MergeFolds_data.pickle.gz')
	return 0

def cleanup():
	import gc
	del globals()['parsed_sentences']
	del globals()['multi_tags_sentences']
	del globals()['parser']
	del globals()['working_dir']
	gc.collect()

def train(ambiguity, nr_iter):
	import random
	from ap import is_projective
	from lib.MaxEntMarkovModel import MaxEntMarkovModel

	print "Training Parser, ambiguity:", ambiguity

	for itn in range(nr_iter):
		correct = 0; total = 0
		random.shuffle(parsed_sentences)
		for i, (words, gold_tags, gold_heads) in enumerate(parsed_sentences):
			sent_key = "".join(words)
			if is_projective(gold_heads) and sent_key in multi_tags_sentences:  # filter non-projective trees
				multi_tags = MaxEntMarkovModel.threshold(multi_tags_sentences[sent_key], ambiguity)
				tags = [word_tags.keys()[0] for word_tags in MaxEntMarkovModel.threshold(multi_tags_sentences[sent_key], 1.0)]
				correct += parser.train_one(words, tags, gold_heads, multi_tags)
				total += len(words)
		if total > 0:
			print itn, '%.3f' % (float(correct-2) / total)  # subtract 2 for the padding the parser adds

	print 'Averaging weights'

	parser.model.average_weights()
	parser.save(model_backup_name(working_dir, ambiguity))


class ParsersTrainDis(MachineLearningModule):
	""" MEMM  multi-tagger into training of Perceptron Parser
	"""
	def run(self, previous):
		from dispy.httpd import DispyHTTPServer

		reg = self.get('regularization')
		func = functools.partial(setup, self.get('uni_dep_base'), self.get('training_file'), self.dir('working'), reg)  # make setup function with some parameters
		cluster = dispy.JobCluster(train, setup=func, cleanup=cleanup, reentrant=True)
		http_server = DispyHTTPServer(cluster) # monitor cluster(s) at http://localhost:8181
		jobs = []
		for i, ambiguity in enumerate([round(0.1 * k, 1) for k in range(11)]):  # i.e. 0.0 to 0.9
			job = cluster.submit(ambiguity, 15)
			job.id = i
			jobs.append(job)
		cluster.wait() # wait for all jobs to finish
		cluster.stats()
		cluster.close()
		http_server.shutdown()

		return True

	def save(self, exp, save_path = None):
		pass

	def load(self, save_path = None, _ = ''):
		pass

def model_backup_name(working_dir, ambiguity):
	return "%s/AmbiguousParser-ambiguity_%.1f.pickle" % (working_dir, ambiguity)

def merge_folds(working_dir, reg):
	num_folds = 10  # 10 fold cross validation
	data = {}
	for i in range(num_folds):
		file_path = backup_file_path(working_dir, i, reg)
		if os.path.exists(file_path + ".gz"):
			data.update(MachineLearningModule.restore(file_path))
		else:
			raise Exception("Fold %d data missing!" % i)
	return data