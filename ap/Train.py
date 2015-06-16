import random
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import Parser as PerceptronParser, DefaultList, PerceptronTagger
from lib.measure import POSTaggerMeasure

class Train(MachineLearningModule):
	""" Train Perceptron on our own data so the results are comparable.
	"""
	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.model = None

	def run(self, tagger_model):

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
			# Do the tagging once as it takes a long time with the MEMM tagger
			tin_tags = {}
			tester = POSTaggerMeasure(pparser.tagger.get_classes())
			for dep_tree in sentences:
				if is_projective(dep_tree):
					words, gold_tags, gold_heads, _ = honnibalify(dep_tree)
					tags = pparser.tagger.tag(words)
					tin_tags["".join(words)] = tags
					tester.test(words,tags,gold_tags,True)

			# DO the parser's online training iterations
			for itn in range(num_iter):
				correct = 0; total = 0; skipped = 0
				random.shuffle(sentences)
				for dep_tree in sentences:
					if is_projective(dep_tree):  # filter non-projective trees
						words, gold_tags, gold_heads, _ = honnibalify(dep_tree)
						tags = tin_tags["".join(words)]  # our (tin-)tags instead of the gold-tags
						#correct += pparser.train_one(itn, words, gold_tags, gold_heads)
						pad_tokens(words); pad_tokens(tags)  #add beginning and end crap to words & tags
						correct += pparser.train_one(itn, words, tags, gold_heads)
						total += len(gold_heads)
					else:
						skipped += 1
				print "Iteration #", itn, "Correct", round(float(correct) / total * 100, 2), "%"
			print "Non-Projective", round(float(skipped)/len(sentences)  * 100, 2), "%"


		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		parsed_sentences = data.parsed_sents(self.config('training_file'))


		print "Train the Parser"
		parser = PerceptronParser(tagger=tagger_model.tagger, load=False)
		train(parser, parsed_sentences, num_iter=15)
		parser.save(save_dir=self.working_dir())

		return False
