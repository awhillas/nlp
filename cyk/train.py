"""
CYK Training
"""

from Counts import Counts
from lib.ml_framework import MachineLearningModule


class Train(MachineLearningModule):
	
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.model = Counts()
	
	def run(self, last):
		# Training
		print "Training model..."
		self.model.count_trees(self.config("training_file"), 2)
		return True