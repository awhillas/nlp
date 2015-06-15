import random
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import Parser as PerceptronParser, DefaultList, PerceptronTagger
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

		def honnibalify(deptree):
			# Oh, for an abstraction to the data...!
			words = DefaultList('')
			tags = DefaultList('')
			heads = [None]
			labels = [None]
			for i, n in deptree.nodes.iteritems():
				if n['word'] is not None:
					words.append(n['word'])  # TODO: perhaps intern(n['word'].encode('utf8')) would speed things up but this need to be consistent and do it in the tagger too :-?220
					tags.append(n['tag'])    #  same here!
					heads.append(n['head'] if n['head'] != 0 else len(deptree.nodes))  # ROOT moved to the end
					labels.append(n['rel'])
			pad_tokens(words); pad_tokens(tags)
			return words, tags, heads, labels

		def is_projective(dep_tree):
				length = len(dep_tree.nodes)
				heads = [n['head'] for n in dep_tree.nodes.itervalues()]
				for w1 in range(length):
					if heads[w1] is not None:
						h1 = heads[w1]
						for w2 in range(length):
							if heads[w2] is not None and arcs_cross(w1, h1, w2, heads[w2]):
								return False
				return True

		def arcs_cross(w1, h1, w2, h2):
			if w1 > h1:
				w1, h1 = h1, w1
			if w2 > h2:
				w2, h2 = h2, w2
			if w1 > w2:
				w1, h1, w2, h2 = w2, h2, w1, h1
			return w1 < w2 < h1 < h2 or w1 < w2 == h2 < h1

		def train(pparser, sentences, num_iter):
			for itn in range(num_iter):
				correct = 0; total = 0; skipped = 0
				random.shuffle(sentences)
				for dep_tree in sentences:
					if is_projective(dep_tree):  # filter non-projective trees
						words, gold_tags, gold_heads, _ = honnibalify(dep_tree)
						correct += pparser.train_one(itn, words, gold_tags, gold_heads)
						total += len(gold_heads)
					else:
						skipped += 1
				print "Iteration #", itn, "Correct", round(float(correct) / total * 100, 2), "%"
			print "Non-Projective", round(float(skipped)/len(sentences)  * 100, 2), "%"

		def train_ptron_tagger(tagger, sentences, nr_iter=5):
			tagger.start_training(sentences)
			for itn in range(nr_iter):
				random.shuffle(list(sentences))
				for s in sentences:
					words, gold_tags = zip(*s)
					tagger.train_one(words, gold_tags)
				tagger.model.average_weights()
			return tagger

		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		tagged_sentences = data.tagged_sents(self.config('training_file'))
		parsed_sentences = data.parsed_sents(self.config('training_file'))

		# print "Loading/Training the MaxEnt tagger"
		# tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		# if not tagger.load(self.working_dir()):
		# 	tagger.train(tagged_sentences)
		print "Loading/Training the Preceptron tagger"
		tagger = PerceptronTagger(load=False, save_dir=self.working_dir())
		if not tagger.load():
			print "training"
			tagger = train_ptron_tagger(tagger, tagged_sentences)
		tagger.save()

		print "Train the Parser"
		parser = PerceptronParser(tagger=tagger, load=False)
		train(parser, parsed_sentences, num_iter=15)
		parser.save(save_dir=self.working_dir())

		return False
