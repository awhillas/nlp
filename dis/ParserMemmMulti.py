import random
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import AmbiguousParser
from ap import is_projective, dep_tree_to_list
from lib.MaxEntMarkovModel import MaxEntMarkovModel
from lib.csv_logger import CSVLogger

class ParserMemmMulti(MachineLearningModule):
	""" MEMM  multi-tagger into training of Perceptron Parser
	"""
	PREVIOUS_MODULE = 'dis.MemmMultiTag'

	def run(self, previous):

		def train(pparser, multi_tags_sentences, sentences, ambiguity, nr_iter=3):
			print "Training Parser"
			for itn in range(nr_iter):
				correct = 0; total = 0
				random.shuffle(sentences)
				for i, (words, gold_tags, gold_heads) in enumerate(sentences):
					if is_projective(gold_heads):  # filter non-projective trees
						multi_tags = MaxEntMarkovModel.threshold(multi_tags_sentences[i], ambiguity)
						tags = MaxEntMarkovModel.threshold(multi_tags, 1.0)  # ambiguity = 1.0 == no ambiguity == 1-best tag
						correct += pparser.train_one(words, tags, gold_heads, multi_tags)
						total += len(words)
				if total > 0:
					print itn, '%.3f' % (float(correct) / float(total))
			print 'Averaging weights'
			pparser.model.average_weights()


		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		parsed_sentences = [(dep_tree_to_list(dep_tree)) for dep_tree in data.parsed_sents(self.get('training_file'))]
		pos = POSTaggerMeasure(tagger.get_classes())
		uas = UASMeasure()
		logger = CSVLogger(self.dir('output') + "/Parser_Multi-Tags.log.csv", pos.cols() + uas.cols() + self.cols() + ['ambiguity'])


		for ambiguity in [round(0.1 * i, 1) for i in range(10)]:  # i.e. 0.0 to 0.9
			self.log('ambiguity', ambiguity)

			print "Start Parser training..."

			self.tagged = previous.tagged
			self.parser = AmbiguousParser(load=False, save_dir=self.working_dir())
			train(self.parser, multi_tags, parsed_sentences, ambiguity, 15)
			self.save(self.dir('working'), '-ambiguity_%.1f'%ambiguity)

			print "Testing..."

			for i, sentence in enumerate(parsed_sentences):
				words, gold_tags, gold_heads = dep_tree_to_list(sentence)
				if is_projective(gold_heads):  # filter non-projective trees
					print 'Testing', i, 'of', len(parsed_sentences)
					# Do the business
					tags = tagger.tag(words)
					heads = parser.parse(words, tags)
					# Measure it
					uas.test(heads, gold_heads)
					pos.test(words, tags, gold_tags)
			# Show results
			pos.totals()
			uas.totals()

		return True

	def save(self, path = None, _ = ''):
		self.parser.save(save_dir=path)

	def load(self, path = None, _ = ''):
		self.parser = AmbiguousParser(load=True, save_dir=path)

