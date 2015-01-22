
class PosTest(MachineLearningModule):
	
	input_module = 'cyk.parse'
	
	def run(self, parser):
		gold_standard = open(self.config.get(self.data_set_id, "testing_tags_file"), "rU")
		for tree in parser.results:
			# Get just the POS for the
			result = [pos for word, pos in tree.pos()]
			target = gold_standard.readline().split(' ')
			if len(result) == len(target):
				errors = len([i for i, j in zip(result, target) if i != j])
			elif len(result) < len(target):
			else:	# target shorter..
				pass
