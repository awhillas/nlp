#! /usr/bin/python

__author__="Alexander Whillas <whillas@gmail.com>"
__date__ ="$Jan 2015"


import sys, json, pprint
from pcfg import PCFG
import ConfigParser

import MyTree


def main(config, data_set_id, model):
	# Training
	# print "Training model..."
	#
	# model = Counts()
	# i = 0
	# input = open(config.get(data_set_id, "training_file"), "rU")
	# for sentence in input:
	# 	i += 1
	# 	if i %100 == 0:
	# 		print i
	# 	model.nltk_count(sentence)

	# Test: use to parsed sentences
	print "Testing..."

	with CYK(PCFG(model)) as parser:
		output = open(config.get(data_set_id, "output") + "/cyk_bnc_init.results.txt", "w+")
		for line in open(config.get(data_set_id, "testing_file")):
			print line
			result = parser.parse(line, "TOP")	# Parse the sentence
			if len(result) > 0:
				# pprint.pprint(result)
				resultTree = MyTree.from_list(result)
				print resultTree
				
				# Get just the POS for the
				pos = " ".join([pos for word, pos in resultTree.pos()]) + "\n"
				output.write(pos)
				print "\n"+pos

				resultTree.un_chomsky_normal_form()
				print resultTree

		output.close()

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print """
			Usage:
			cyk_parse.py <training_file> <testing_file>
			Where:
				training_file - Penn treebank stye file. One tree per line.
				testing_file - raw sentences for producing trees of.
			Both path should be relative to the data/input folder in specified in ml_config.ini
		"""
		sys.exit(1)
	main(sys.argv[1], sys.argv[2])
