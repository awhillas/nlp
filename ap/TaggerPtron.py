import random
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import PerceptronTagger

class TaggerPtron(MachineLearningModule):
	""" Train Perceptron Tagger for use with the Perceptron Parser
	"""
	def run(self, previous):

		def train(tagger, sentences, nr_iter=5):
			tagger.start_training(sentences)
			for itn in range(nr_iter):
				random.shuffle(list(sentences))
				for s in sentences:
					words, gold_tags = zip(*s)
					tagger.train_one(words, gold_tags)
				tagger.model.average_weights()

		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		tagged_sentences = data.tagged_sents(self.get('training_file'))
		tagger = PerceptronTagger(load=False, save_dir=self.working_dir())

		print "Preceptron tagger"
		if not tagger.load():
			print "...training."
			train(tagger, tagged_sentences)

		self.keepers['tagger'] = tagger
		if not self._experiment.no_cache:
			tagger.save()

		return True
