from ml_framework import MachineLearningModule
from measure import Measure
from tabulate import tabulate

class PosTest(MachineLearningModule):

	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'cyk.Parse'
	
	def run(self, parser):
		m = Measure()
		gold_standard = open(self.config.get(self.data_id, "testing_tags_file"), "rU")
		for tree in parser.results:
			# Get just the POS for the
			result = [pos for word, pos in tree.pos()]
			words = [word for word, pos in tree.pos()]
			target = gold_standard.readline().split(' ')
			errors = 0
			if len(result) == len(target):
				pos_tags = zip(result, target)
				errors = len([i for i, j in pos_tags if i.strip() != j.strip()])
			else:  # target shorter..
				errors = len(target)  # Count the whole thing as an error :(
				print "Error: Results wrong length: ", len(result), "vs.", len(target)
				pass
			m.tp(len(words) - errors)
			m.fp(errors)
			print tabulate([["RESULT:"]+result, ["GOLD:"]+target], [""]+words, tablefmt="pipe")
			print "Errors: ", errors, "\n"
		print m
		return True