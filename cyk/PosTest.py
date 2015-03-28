import pandas as pd
from nltk.tree import Tree

from lib.ml_framework import MachineLearningModule
from lib.measure import Measure
from tabulate import tabulate


class PosTest(MachineLearningModule):

	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'cyk.Parse'
	
	def run(self, parser):
		"""
		TODO: More detail in the error break down i.e. errors by POS tag
		TODO: tree structure error alla M.Collins
		"""
		def tree_to_pos(t):
			p = [pos for word, pos in t.pos()]
			w = [word for word, pos in t.pos()]
			return (w, p)

		m = Measure()
		tag_error_count = {}
		gold_standard = open(self.config("testing_trees"), "rU")

		for tree in parser.results:
			gold_tree = Tree.fromstring(gold_standard.readline())
			if tree is None:
				print "Missing a tree??"
				print " ".join(gold_tree.flatten())
				continue

			# Get just the POS for the
			words, pos = tree_to_pos(tree)
			_, gold = tree_to_pos(gold_tree)

			errors = 0
			if len(pos) == len(gold):
				pos_tags = zip(pos, gold)
				#errors = len([i for i, j in pos_tags if i.strip() != j.strip()])
				for r, t in pos_tags:
					if r.strip() != t.strip():
						errors += 1
						tag_error_count.setdefault(t, {})
						tag_error_count[t].setdefault(r, 0)
						tag_error_count[t][r] += 1
				m.tp(len(words) - errors)  # true positives
				m.fp(errors)  # false positives
				print tabulate([["RESULT:"]+pos, ["GOLD:"]+gold], [""]+words, tablefmt="pipe")
				print "Errors: ", errors, "\n"
			else:  # target shorter..
				errors = len(gold)  # Count the whole thing as an error :(
				print "Error: Results wrong length: ", len(pos), "vs.", len(gold)
				print "Output: ", tree
				print "Actual:", gold_tree
				pass

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