import random
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import AmbiguousParser
from ap import is_projective, dep_tree_to_list
from lib.MaxEntMarkovModel import MaxEntMarkovModel
from lib.csv_logger import CSVLogger
from lib.measure import UASMeasure, POSTaggerMeasure
from dis.MemmTag import tagged_file_name
from dis.ParsersTrain import model_backup_name

class Testing(MachineLearningModule):
	""" MEMM  multi-tagger into training of Perceptron Parser
	"""
	# PREVIOUS_MODULE = 'dis.ParserMemmMulti'

	def run(self, previous):

		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		parsed_sentences = [(dep_tree_to_list(dep_tree)) for dep_tree in data.parsed_sents(self.get('testing_file'))]
		memm_multi_tags = self.restore(self.dir('working')+"/tagged_sentences_testing_file_multi-reg_0.66.pickle")
		tag_set = self.restore(self.dir('working') + "/tags.pickle")
		pos = POSTaggerMeasure(tag_set)
		uas = UASMeasure()
		logger = CSVLogger(self.dir('output') + "/Parser_Multi-Tags.log.csv", pos.cols() + uas.cols() + self.cols()
						   + ['ambiguity', 'tags/word', 'projective', 'skipped'])

		for ambiguity in [round(0.1 * k, 1) for k in range(11)]:  # i.e. 0.0 to 0.9
			self.log('ambiguity', ambiguity)
			parser = AmbiguousParser(model_backup_name(self.dir('working'), ambiguity))

			print "Testing parser...", "ambiguity:", ambiguity

			total = len(parsed_sentences)
			total_words = 0
			total_tags = 0
			projective_count = 0
			for i, sentence in enumerate(parsed_sentences, start=1):
				words, gold_tags, gold_heads = sentence
				sent_key = "".join(words)
				if sent_key in memm_multi_tags:  # only if we managed to
					# print 'Testing', i, 'of', len(parsed_sentences)
					multi_tags = MaxEntMarkovModel.threshold(memm_multi_tags[sent_key], ambiguity)
					tags = [tag_set.keys()[0] for tag_set in MaxEntMarkovModel.threshold(memm_multi_tags[sent_key], 1.0)]
					# Do the business
					heads = parser.parse(words, tags, multi_tags)
					# Measure it
					total_words += len(words)
					total_tags += sum([len(ts) for ts in multi_tags])
					uas.test(heads, gold_heads)
					pos.test(words, tags, gold_tags, verbose=False)
					if is_projective(gold_heads):
						projective_count += 1
			self.log('skipped', per(total-len(parsed_sentences), total))
			self.log('projective', per(projective_count,total))
			self.log('tags/word', round(float(total_tags)/total_words, 2))
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
		pass

	def load(self, save_path = None, _ = ''):
		# self.parser = AmbiguousParser(load=True, save_dir=save_path)
		pass

def per(top, total):
	return round(float(top) / total * 100, 2)