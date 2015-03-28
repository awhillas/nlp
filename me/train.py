"""
MaxEnt Markov Model training
"""

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, HonibbalsFeats

class Train(MachineLearningModule):

	# def __init__(self, config, data_set_id):
	# 	MachineLearningModule.__init__(self, config, data_set_id)

	def run(self, last):
		# Training
		print "Training MEMM..."

		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus

		model = MaxEntMarkovModel.train(data, HonibbalsFeats)