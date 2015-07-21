import random, os
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import AmbiguousParser
from ap import is_projective, dep_tree_to_list
from lib.MaxEntMarkovModel import MaxEntMarkovModel
from dis.MemmTag import tagged_file_name
from dis.MemmMultiTag import backup_file_path


def train(pparser, multi_tags_sentences, sentences, ambiguity, nr_iter=3):
	print "Training Parser"
	for itn in range(nr_iter):
		correct = 0; total = 0
		random.shuffle(sentences)
		for i, (words, gold_tags, gold_heads) in enumerate(sentences):
			sent_key = "".join(words)
			if is_projective(gold_heads) and sent_key in multi_tags_sentences:  # filter non-projective trees
				multi_tags = MaxEntMarkovModel.threshold(multi_tags_sentences[sent_key], ambiguity)
				tags = [word_tags.keys()[0] for word_tags in MaxEntMarkovModel.threshold(multi_tags_sentences[sent_key], 1.0)]
				correct += pparser.train_one(words, tags, gold_heads, multi_tags)
				total += len(words)
		if total > 0:
			print itn, '%.3f' % (float(correct-2) / total)  # subtract 2 for the padding the parser adds
	print 'Averaging weights'
	pparser.model.average_weights()


class ParsersTrain(MachineLearningModule):
	""" MEMM  multi-tagger into training of Perceptron Parser
	"""
	# PREVIOUS_MODULE = 'dis.MemmMultiTag'

	def run(self, previous):
		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		parsed_sentences = [(dep_tree_to_list(dep_tree)) for dep_tree in data.parsed_sents(self.get('training_file'))]
		memm_multi_tags = merge_folds(self.dir('working'), self.get('regularization'))

		for ambiguity in [round(0.1 * k, 1) for k in range(11)]:  # i.e. 0.0 to 1.0
			self.log('ambiguity', ambiguity)

			print "Start Parser training..."

			self.parser = AmbiguousParser()
			train(self.parser, memm_multi_tags, parsed_sentences, ambiguity, 15)
			self.parser.save(model_backup_name(self.dir('working'), ambiguity))

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