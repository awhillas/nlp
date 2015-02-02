from ml_framework import MachineLearningModule
from cyk import CYK
from pcfg import PCFG
from MyTree import MyTree


class Parse(MachineLearningModule):

	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'cyk.Train'
		self.results = []
	
	def run(self, trained):
		with CYK(PCFG(trained.model)) as parser:
			print "Parsing..."
			#output = open(config.get(data_set_id, "output") + "/cyk_bnc_init.results.txt", "w+")
			i = 1
			for line in open(self.config.get(self.data_id, "testing_file")):
				print '#', i, ': ', line, '\n'
				result = parser.parse(line, "TOP")  # Parse the sentence
				if len(result) > 0:
					# pprint.pprint(result)
					result_tree = MyTree.from_list(result)
					# Transform back from CNF
					result_tree.un_chomsky_normal_form()
					print result_tree
					self.results.append(result_tree)
				else:
					self.results.append(None)
				i += 1
		return True