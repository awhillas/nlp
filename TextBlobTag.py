""" POS tagger using the TextBlob library
	requires:
		pip install -U textblob
		pip install -U textblob-aptagger
	Using predominantly for bench marking
"""

from __future__ import print_function
from ml_framework import MachineLearningModule
from textblob import TextBlob
from textblob_aptagger import PerceptronTagger


class TextBlobTag(MachineLearningModule):
	def run(self, sentences):
		with open(self.get_output_file_name() + ".tags.txt", 'w') as f:
			for s in sentences:
				blob = TextBlob(s, pos_tagger=PerceptronTagger())
				print(" ".join([tup[1] for tup in blob.tags]), file=f)
		return True
