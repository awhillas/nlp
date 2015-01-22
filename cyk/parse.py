from cyk.cyk import CYK
from cyk.pcfg import PCFG

class Parse(MachineLearningModule):
	
	input_module = 'cyk.train'
	
	def init(self):
		self.results = []
	
	def run(self, model):
		with CYK(PCFG(model)) as parser:
			print "Parsing..."
			#output = open(config.get(data_set_id, "output") + "/cyk_bnc_init.results.txt", "w+")
			i = 1
			for line in open(self.config.get(self.data_set_id, "testing_file")):
				print '#', i, ': ', line, '\n'
				result = parser.parse(line, "TOP")	# Parse the sentence
				if len(result) > 0:
					# pprint.pprint(result)
					resultTree = MyTree.from_list(result)
					# Transform back from CNF
					resultTree.un_chomsky_normal_form()
					print resultTree
					self.results.append(resultTree)
				else:
					self.results.append(None)
				i += 1
			return True