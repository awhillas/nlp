"""
MaxEnt Markov Model training
"""
import time, datetime
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.pos_tagging import tag_all_shared

class PredictShare(MachineLearningModule):
	PREVIOUS_MODULE = 'me.Train'

	def run(self, previous):
		self.tagger = previous.tagger
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus

		# Using the MaxEnt model
		start = time.time()
		tag_all_shared(
			data.sents(self.get('cv_file')),
			tagger=self.tagger.label,
			normaliser=self.tagger.normaliser,
			working_path=self.dir('working'),
			block_size=10,
			output_pickle='/memm_tagged_sentences.pickle'
		)
		self.log("Total Time", str(datetime.timedelta(seconds= (time.time() - start))))

		return False  # Don't save