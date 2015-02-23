""" POS tagger using the TextBlob library
	requires:
		pip install -U textblob
		pip install -U textblob-aptagger
	Using predominantly for bench marking
"""

from __future__ import print_function
from ml_framework import MachineLearningModule
#from textblob import TextBlob
#from textblob_aptagger import PerceptronTagger


class TextblobTag(MachineLearningModule):
	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'tb.TextblobTrain'

	def run(self, tagger):
		with open(self.get_output_file_name(), 'w') as f:
			for sentence in open(self.config.get(self.data_id, "testing_file")):
				tags = tagger.tag(sentence)
				print(" ".join([tup[1] for tup in tags]))
				# print(" ".join([tup[1] for tup in blob.tags]), file=f)

		return True
