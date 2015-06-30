import random
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import Parser as PerceptronParser, PerceptronTagger
from lib.measure import POSTaggerMeasure
from ap import is_projective, dep_tree_to_list

class ParserPtron(MachineLearningModule):
	""" Train Perceptron Tagger+Parser.
	"""
	def run(self, previous):

		def train(pparser, ptagger, sentences, nr_iter=3):
			ptagger.start_training(sentences)
			pos_tester = POSTaggerMeasure(ptagger.classes)
			for itn in range(nr_iter):
				correct = 0; total = 0
				random.shuffle(sentences)
				for words, gold_tags, gold_heads in sentences:
					if is_projective(gold_heads) and len(words) > 1:  # filter non-projective trees
						tags = ptagger.tag(words)
						pos_tester.test(words, tags, gold_tags, verbose=False)
						correct += pparser.train_one(itn, words, tags, gold_heads)
						if itn < 5:
							ptagger.train_one(words, gold_tags)
						total += len(words)
				if total > 0:
					print itn, '%.3f' % (float(correct) / float(total))
				if itn == 4:
					ptagger.model.average_weights()
					pos_tester.totals()
			print 'Averaging weights'
			pparser.model.average_weights()


		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		parsed_sentences = [(dep_tree_to_list(dep_tree)) for dep_tree in data.parsed_sents(self.config('training_file'))]

		print "Train the Parser"

		self.tagger = PerceptronTagger(load=False, save_dir=self.working_dir())
		self.parser = PerceptronParser(load=False, save_dir=self.working_dir())

		train(self.parser, self.tagger, parsed_sentences, 15)

		return True

	def save(self, path = None, filename_prefix = ''):
		self.parser.save()
		self.tagger.save()

	def load(self, path = None, filename_prefix = ''):
		self.tagger = PerceptronTagger(load=True, save_dir=self.working_dir())
		self.parser = PerceptronParser(load=True, save_dir=self.working_dir())

