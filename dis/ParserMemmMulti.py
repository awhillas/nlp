import random
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import AmbiguousParser
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
from ap import is_projective, dep_tree_to_list
import cPickle as pickle
from os.path import exists

class ParserMemmMulti(MachineLearningModule):
	""" MEMM  multi-tagger into training of Perceptron Parser
	"""
	PREVIOUS_MODULE = 'ap.TaggerMemm'
	BACKUP_FILENAME = '/multi-tagged_sentences'

	def run(self, previous):

		def backup(data, path):
			with open(path, 'wb') as f:
				pickle.dump(data, f, 2)
				print "Saved", path

		def restore(path):
			if not exists(path):
				print "Could not load", path
				return False
			else:
				with open(path, 'rb') as f:
					return pickle.load(f)

		def multi_tags_backup_filename(ambiguity):
			return self.working_dir() + ParserMemmMulti.BACKUP_FILENAME + '-ambiguity_%.2f' % ambiguity + '.pickle'

		def train(pparser, tin_tags, sentences, nr_iter=3):
			print "Training Parser"
			for itn in range(nr_iter):
				correct = 0; total = 0
				random.shuffle(sentences)
				for i, (words, gold_tags, gold_heads) in enumerate(sentences):
					if is_projective(gold_heads):  # filter non-projective trees
						tags, multi_tags = tin_tags["".join(words)]  # i.e. tin is not gold
						correct += pparser.train_one(words, tags, gold_heads, multi_tags)
						total += len(words)
				if total > 0:
					print itn, '%.3f' % (float(correct) / float(total))
			print 'Averaging weights'
			pparser.model.average_weights()


		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		parsed_sentences = [(dep_tree_to_list(dep_tree)) for dep_tree in data.parsed_sents(self.config('training_file'))]

		print "Start Parser training..."

		self.tagger = previous.tagger
		self.tagged = previous.tagged
		self.parser = AmbiguousParser(load=False, save_dir=self.working_dir())

		train(self.parser, self.tagged, parsed_sentences, 15)

		return True

	def save(self, path = None, filename_prefix = ''):
		self.parser.save(save_dir=self.working_dir())
		# self.tagger.save(save_dir=self.working_dir())

	def load(self, path = None, filename_prefix = ''):
		self.tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		self.tagger.load(self.working_dir(), filename_prefix='-reg_%.2f' % self.get('regularization'))
		self.parser = AmbiguousParser(load=True, save_dir=self.working_dir())

