import random
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import Parser as PerceptronParser, PerceptronTagger
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation
from ap import is_projective, dep_tree_to_list


class ParserMemm(MachineLearningModule):
	""" Train Perceptron on our own data so the results are comparable.
	"""
	PREVIOUS_MODULE = 'ap.TaggerMemm'

	def run(self, previous):

		def train(pparser, sentences, nr_iter=3):
			for itn in range(nr_iter):
				correct = 0; total = 0
				random.shuffle(sentences)
				for words, gold_tags, gold_heads in sentences:
					if is_projective(gold_heads) and len(words) > 1:  # filter non-projective trees
						# TODO: should use cross-fold train for the tagger
						# 	and keep the predictions for the parser training
						# 	use the CV set as well.
						correct += pparser.train_one(itn, words, gold_tags, gold_heads)
						total += len(words)
				if total > 0:
					print itn, '%.3f' % (float(correct) / float(total))
			print 'Averaging weights'
			pparser.model.average_weights()


		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		parsed_sentences = [(dep_tree_to_list(dep_tree)) for dep_tree in data.parsed_sents(self.config('training_file'))]

		print "Train the Parser"

		self.tagger = previous.tagger
		self.parser = PerceptronParser(load=False, save_dir=self.working_dir())

		train(self.parser, parsed_sentences, 15)

		return True

	def save(self, path = None, filename_prefix = ''):
		self.parser.save(save_dir=self.working_dir())
		# self.tagger.save(save_dir=self.working_dir())

	def load(self, path = None, filename_prefix = ''):
		self.tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		self.tagger.load(save_dir=self.working_dir())
		self.parser = PerceptronParser(load=True, save_dir=self.working_dir())

