import random
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.PerceptronParser import PerceptronTagger

class PTagger(MachineLearningModule):
	""" Train Perceptron Tagger for use with the Perceptron Parser
	"""
	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.tagger = PerceptronTagger(load=False, save_dir=self.working_dir())

	def run(self, _=None):

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

		if not self.tagger.load():
			print "Training the Preceptron tagger"
			self.tagger = train_ptron_tagger(self.tagger, tagged_sentences)
		else:
			print "Loaded the Preceptron tagger"

		if not self._experiment.no_cache:
			self.tagger.save()

		return True
