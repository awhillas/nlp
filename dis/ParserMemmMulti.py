import random
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import AmbiguousParser
from ap import is_projective, dep_tree_to_list
from lib.MaxEntMarkovModel import MaxEntMarkovModel
from lib.csv_logger import CSVLogger
from lib.measure import UASMeasure, POSTaggerMeasure

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
					sent_key = "".join(words)
					if is_projective(gold_heads) and sent_key in multi_tags_sentences:  # filter non-projective trees
						multi_tags = MaxEntMarkovModel.threshold(multi_tags_sentences[sent_key], ambiguity)
						tags = [word_tags.keys()[0] for word_tags in MaxEntMarkovModel.threshold(multi_tags_sentences[sent_key], 1.0)]
						correct += pparser.train_one(words, tags, gold_heads, multi_tags)
						total += len(words)
				if total > 0:
					print itn, '%.3f' % (float(correct) / float(total))
			print 'Averaging weights'
			pparser.model.average_weights()


		# Get (words, tags) sequences for all sentences
		multi_tags = previous.mult_tagged
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		parsed_sentences = [(dep_tree_to_list(dep_tree)) for dep_tree in data.parsed_sents(self.get('training_file'))]

		pos = POSTaggerMeasure(previous.all_tags)
		uas = UASMeasure()
		logger = CSVLogger(self.dir('output') + "/Parser_Multi-Tags.log.csv", pos.cols() + uas.cols() + self.cols() + ['ambiguity', 'skipped'])

		for ambiguity in [round(0.1 * k, 1) for k in range(10)]:  # i.e. 0.0 to 0.9
			self.log('ambiguity', ambiguity)

			print "Start Parser training..."

			self.parser = AmbiguousParser(load=False, save_dir=self.working_dir())
			train(self.parser, multi_tags, parsed_sentences, ambiguity, 15)
			self.save(self.dir('working'), '-ambiguity_%.1f'%ambiguity)

			print "Testing..."

			skipped = 0
			for i, sentence in enumerate(parsed_sentences):
				words, gold_tags, gold_heads = sentence
				sent_key = "".join(words)
				if is_projective(gold_heads) and sent_key in multi_tags:  # filter non-projective trees
					print 'Testing', i, 'of', len(parsed_sentences)
					# Do the business
					tags = [word_tags.keys()[0] for word_tags in MaxEntMarkovModel.threshold(multi_tags[sent_key], 1.0)]
					heads = self.parser.parse(words, tags, multi_tags[sent_key])
					# Measure it
					uas.test(heads, gold_heads)
					pos.test(words, tags, gold_tags)
				else:
					skipped += 1
			self.log('skipped', skipped)
			# Show results
			pos.totals()
			uas.totals()

			if not self.get('no_log'):
				data = pos.log()
				data.update(uas.log())
				data.update(self.get_log())
				logger.add(**data)

		return True

	def save(self, save_path = None, _ = ''):
		self.parser.save(save_dir=save_path)

	def load(self, save_path = None, _ = ''):
		self.parser = AmbiguousParser(load=True, save_dir=save_path)
