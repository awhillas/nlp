from ml_framework import MachineLearningModule


class PosTest(MachineLearningModule):

	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'cyk.Parse'
	
	def run(self, parser):
		gold_standard = open(self.config.get(self.data_id, "testing_tags_file"), "rU")
		for tree in parser.results:
			# Get just the POS for the
			result = [pos for word, pos in tree.pos()]
			target = gold_standard.readline().split(' ')
			if len(result) == len(target):
				errors = len([i for i, j in zip(result, target) if i != j])
			elif len(result) < len(target):
				pass
			else:  # target shorter..
				pass
		return True