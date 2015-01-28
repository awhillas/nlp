from Counts import Counts
from ml_framework import MachineLearningModule


class Train(MachineLearningModule):
	
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.model = Counts()
	
	def run(self, last):
		# Training
		print "Training model..."
		i = 0
		data = open(self.config.get(self.data_id, "training_file"), "rU")
		for sentence in data:
			i += 1
			if i % 100 == 0:
				print i
			self.model.nltk_count(sentence)
		return True