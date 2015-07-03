"""
MaxEnt Markov Model training
"""
import time, datetime
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.pos_tagging import tag_all

class Predict(MachineLearningModule):
	PREVIOUS_MODULE = 'me.Train'
	BACKUP_FILENAME = '/memm_labeled_sequences'

	def run(self, previous):
		self.tagger = previous.tagger
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus

		# Using the MaxEnt model
		start = time.time()
		self.labeled_sequences = tag_all(
			data.sents(self.get('cv_file')),
			tagger=self.tagger.label,
			normaliser=self.tagger.normaliser
		)
		self.log("Total Time", str(datetime.timedelta(seconds= (time.time() - start))))
		return True