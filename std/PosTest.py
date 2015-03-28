from lib.ml_framework import MachineLearningModule
from lib.measure import Measure
#from tabulate import tabulate
import pandas as pd


class PosTest(MachineLearningModule):
	"""
	Generic POS testing script.
	Assumptions:
		- I the config in the data section there is an entry for output_file which is the name of the output file to
		  test in the current working_dir()
		- There is a 'gold_output' entry in the the config which is the gold standard POS tags to compare with.
	"""

	def run(self, parser):
		m = Measure()
		tag_error_count = {}

		with open(self.working_dir()+'/'+self.config('output_file'), 'r') as auto_file:
			with open(self.working_dir()+'/'+self.config('gold_output'), 'r') as gold_file:

				for gold_line in gold_file:
					gold = gold_line.strip().split(" ")
					pos = auto_file.readline().strip().split(" ")

					for i, gold_tag in enumerate(gold):
						errors = 0
						if len(pos) > i:
							auto_tag = pos[i]
							if gold_tag.strip() != auto_tag.strip():
								errors += 1
								tag_error_count.setdefault(gold_tag, {})
								tag_error_count[gold_tag].setdefault(auto_tag, 0)
								tag_error_count[gold_tag][auto_tag] += 1
								m.fp()	# False positive???
							else:
								m.tp()  # true positives
					#print tabulate([["RESULT:"]+pos, ["GOLD:"]+gold], [""]+words, tablefmt="pipe")

			# POS Tag Confusion Matrix

			df = pd.DataFrame(data=tag_error_count)
			df['totals'] = df.sum(axis=1)
			df.fillna('').sort(columns='totals').to_csv(self.output_dir()+"/pos_tag_confusion_matrix.csv")
			print "POS Tag Confusion Matrix"
			pd.options.display.width = 500
			pd.options.display.max_columns = 50
			print df.fillna('').sort(columns='totals')

			print m
			return True