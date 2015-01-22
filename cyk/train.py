from Counts import Counts
from ml_framework import MachineLearningModule

class Train(MachineLearningModule):
	
	def init(self):
		self.model = Counts()
	
	def run():
		# Training
		print "Training model..."
		i = 0
		input = open(self.config.get(self.data_set_id, "training_file"), "rU")
		for sentence in input:
			i += 1
			if i %100 == 0:
				print i
			self.model.nltk_count(sentence)