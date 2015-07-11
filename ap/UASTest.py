from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.measure import UASMeasure, POSTaggerMeasure
from lib.csv_logger import CSVLogger
from ap import dep_tree_to_list, is_projective

class UASTest(MachineLearningModule):
	""" Test the Parser + Tagger on the test set.
	"""
	# PREVIOUS_MODULE = 'ap.TrainParser'

	def run(self, previous):
		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		parsed_sentences = data.parsed_sents(self.get('testing_file'))
		ambiguity = self.log_me('Ambiguity', '-')

		tagger = previous.tagger
		parser = previous.parser
		pos = POSTaggerMeasure(tagger.get_classes())
		uas = UASMeasure()
		logger = CSVLogger(self.get('output') + "/Parser-tagger.log.csv", pos.cols() + uas.cols() + list(reversed(self.cols())))

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
		print "\n"

		if not self._experiment.no_log:
			data = pos.log()
			data.update(uas.log())
			data.update(self._experiment.log)
			logger.add(**data)

		return False  # don't save

	def save(self, data, path, filename_prefix = ''):
		pass

	def load(self, path, filename_prefix = ''):
		pass