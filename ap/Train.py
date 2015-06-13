import random
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import Parser as PerceptronParser, DefaultList
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

class Train(MachineLearningModule):
	""" Train Textblob Perceptron on our own data so the results are comparable.
	"""
	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.model = None

	def run(self, _=None):

		def pad_tokens(tokens):
			tokens.insert(0, '<start>')  # TODO: these need to match the taggers start/end
			tokens.append('ROOT')

		def honnibalafy(sentences):
			# Adjust to Matthew Honnibal's unique way of handling data :-/
			words = DefaultList(''); tags = DefaultList('')
			heads = [None]; labels = [None]
			for deptree in sentences:
				for n in deptree.nodelist:
					words.append(intern(n['word']))
					tags.append(intern(n['tag']))
					if 'head' in n:
						heads.append(n['head'] if n['head'] != 0 else len(deptree.nodelist) + 1)  # ROOT moved to the end
					labels.append(n['rel'])
				pad_tokens(words); pad_tokens(tags)
				yield words, tags, heads, labels

		def train(parser, parsed_sentences, nr_iter):
			for itn in range(nr_iter):
				correct = 0; total = 0
				random.shuffle(parsed_sentences)
				for words, gold_tags, gold_heads, gold_label in parsed_sentences:
					correct += parser.train_one(itn, words, gold_tags, gold_heads)


		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		tagged_sentences = data.tagged_sents(self.config('training_file'))
		parsed_sentences = honnibalafy(data.parsed_sents(self.config('training_file')))

		# Use the MaxEnt tagger
		tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		tagger.train(tagged_sentences)

		# Train the model
		parser = PerceptronParser(tagger=tagger, load=False)
		train(parser, parsed_sentences, nr_iter=15)
		parser.save(save_dir=self.working_dir())

		return True