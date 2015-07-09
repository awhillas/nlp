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

		def train(pparser, tin_tags, sentences, nr_iter=3):
			print "Training Parser"
			for itn in range(nr_iter):
				correct = 0; total = 0
				random.shuffle(sentences)
				for i, (words, gold_tags, gold_heads) in enumerate(sentences):
					if is_projective(gold_heads):  # filter non-projective trees
						tags, multi_tags = tin_tags["".join(words)]
						correct += pparser.train_one(words, tags, gold_heads, multi_tags)
						total += len(words)
				if total > 0:
					print itn, '%.3f' % (float(correct) / float(total))
			print 'Averaging weights'
			pparser.model.average_weights()


		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		parsed_sentences = [(dep_tree_to_list(dep_tree)) for dep_tree in data.parsed_sents(self.config('training_file'))]


		for ambiguity in [round(0.1 * i, 1) for i in range(10)]:  # i.e. 0.0 to 0.9
			print "Start Parser training..."
			self.log('ambiguity', ambiguity)
			self.tagged = previous.tagged
			self.parser = AmbiguousParser(load=False, save_dir=self.working_dir())
			tin_tags = MaxEntMarkovModel.threshold(self.tagged, ambiguity)  # i.e. tin is not gold :>
			train(self.parser, tin_tags, parsed_sentences, 15)
			self.save(self.dir('working'), '-ambiguity_%.1f'%ambiguity)

			# Testing

			print "Testing..."
			pos = POSTaggerMeasure(tagger.get_classes())
			uas = UASMeasure()
			logger = CSVLogger(self.dir('output') + "/Parser_Multi-Tags.log.csv", pos.cols() + uas.cols() + list(reversed(self.cols())))


		return True

	def save(self, path = None, _ = ''):
		self.parser.save(save_dir=path)

	def load(self, path = None, _ = ''):
		self.parser = AmbiguousParser(load=True, save_dir=path)

