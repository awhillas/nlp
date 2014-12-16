#! /usr/bin/python

__author__="Alexander Whillas <whillas@gmail.com>"
__date__ ="$Nov 6, 2014"


import sys, operator, json, pprint
from Counts import Counts
from pcfg import PCFG
from json.encoder import JSONEncoder
from ml_config import options
from nltk.tree import Tree

class CYK:
	""" Implementation of the classic CYK parsing algorithm
	"""
	def __init__(self, pcfg):
		self.pcfg = pcfg
		self.f = open("cyk_trouble_sents.txt",'w')

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		""" Like a destructor. Need to use this class like:
		```
		with CYK() as parser:
			# use parser object and __exit__() will be auto. called.
		```
		"""
		self.f.close()

	def parse(self, sentence_string, start_pos):
		return self.cyk(self.chunk(sentence_string), start_pos)

	def write_file(self, line):
		self.f.write(line.encode('latin-1')+"\n")

	def chunk(self, string):
		""" Just chunk on spaces for now. 
			TODO: something more sophisticated.
			return: array of word strings.
		"""
		return string.split()

	def cyk(self, words, start_pos):
		""" Cocke-Kasami-Younger (CKY) Constituency Parsing
			Based on notes from Standford NLP Coursera online course, 2012.
			We don't have unary rules as we are using a strict CNF grammar.
		"""
		n = len(words)	# Number of words in the sentence.
		score = [[{} for _ in range(n + 1)] for _ in range(n + 1)]
		back = [[{} for _ in range(n + 1)] for _ in range(n + 1)]
		for i in range(0, n):
			for A in self.pcfg.getWordTags(words[i]):	# unary non-terminals
				score[i][i+1][A] = self.pcfg.P((A, words[i]))
				back[i][i+1][A] = (0, words[i])
			# handle unarys...
			added = True
			while added:
				added = False
				for A, B in self.pcfg.getUnaryRulesFor(score[i][i+1].keys()):
					if B in score[i][i+1].keys() and score[i][i+1][B] > 0:
						prob = self.pcfg.P((A, B)) * score[i][i+1][B]
						if not A in score[i][i+1].keys() or prob > score[i][i+1][A]:
							score[i][i+1].setdefault(A, 0)
							score[i][i+1][A] = prob
							back[i][i+1].setdefault(A, (0, B))
							back[i][i+1][A] = (0, B)
							added = True
		for span in range(2, n+1):
			for begin in range(0, n - span + 1):
				end = begin + span
				for split in range(begin+1, end):
					rhs_permutations = self.pairPermutations(score[begin][split], score[split][end])
					for (A, B, C) in self.pcfg.lookupRulesFor(rhs_permutations):
						score[begin][end].setdefault(A, 0)
						if B in score[begin][split].keys() and C in score[split][end].keys():
							prob = score[begin][split][B] * score[split][end][C] * self.pcfg.P((A, B, C))
						if prob > score[begin][end][A]:
							score[begin][end][A] = prob
							back[begin][end][A] = (split, B, C)
				# TODO: handle unaries here...
				added = True
				while added:
					added = False
					for A, B in self.pcfg.getUnaryRulesFor(score[begin][end].keys()):
						prob = self.pcfg.P((A, B)) * score[begin][end][B]
						if not A in score[begin][end].keys() or prob > score[begin][end][A]:
							score[begin][end].setdefault(A, prob)
							score[begin][end][A] = prob
							back[begin][end][A] = (0, B)
							added = True

		# if len(back[0][len(back[0])-1]) == 0:
		# 	self.write_file(" ".join(words)) # keep problem sentences for debug
		return self.buildTree(score, back, start_pos)

	def buildTree(self, score, bp, start_pos):
		""" Reconstruct the best parse tree from the give back pointers
		"""
		row = 0
		col = len(bp[0])-1
		if len(score[row][col]):
			if start_pos in score[row][col]:
				root = start_pos
			else:
				root = max(score[row][col].iteritems(), key=operator.itemgetter(1))[0]  # Find the max tree root
			return self.getSubTree(root, (row, col), score, bp)
		else:
			return []

	def getSubTree(self, constituent, position, score, bp):
		""" Rebuild the most probable parse from the score recursively
		:param constituent: most probably parent constituent
		:param position: where in the chart the constituent is
		:param score: table of probabilities built by the CYK algorithm
		:param bp: table of back-pointers of the splits and sub constituents
		:return: dict
		"""
		row = position[0]
		col = position[1]
		if len( bp[row][col][constituent]) == 3:
			split, left, right = bp[row][col][constituent]
			l = self.getSubTree(left, (row, split), score, bp)
			r = self.getSubTree(right, (split, col), score, bp)
			return [constituent, l, r]
		else:
			split, left = bp[row][col][constituent]
			if left != constituent and self.pcfg.isNonTerminal(left):
				return [constituent, self.getSubTree(left, position, score, bp)]
			else:
				return [constituent, left]

	def pretty(self, matrix):
		""" Prints a 2D table nice!
			:param matrix: 2D array to print pretty
		"""
		s = [[str(e) for e in row] for row in matrix]
		lens = [max(map(len, col)) for col in zip(*s)]
		fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
		table = [fmt.format(*row) for row in s]
		print '\n'.join(table)

	def pairPermutations(self, left, right):
		pairs = []
		if len(left) > 0 and len(right) > 0:
			for (l, v) in left.iteritems():
				for (r, v2) in right.iteritems():
					pairs.append((l, r))
		return pairs

class MyTree(Tree):

	@classmethod
	def from_list(cls, lst):
		if len(lst) > 1:
			if isinstance(lst[1], list):   # has children
				if len(lst) == 3:
					return Tree(lst[0], [cls.from_list(lst[1]), cls.from_list(lst[2])])
				elif len(lst) == 2:
					return Tree(lst[0], [cls.from_list(lst[1])])
			else:
				return Tree(lst[0], [lst[1]]) # just a list of children
		else:
			print("Not a binary tree?")
			print lst

	@classmethod
	def to_dict(cls, tree):
		tdict = {}
		for t in tree:
			if isinstance(t, Tree) and isinstance(t[0], Tree):
				tdict[t.node] = cls.to_dict(t)
			elif isinstance(t, Tree):
				tdict[t.node] = t[0]
		return tdict

	@classmethod
	def to_json(cls, dict):
		return json.dumps(cls.to_dict())

def main(training_file, testing_file):

	# Training
	print "Training..."

	counter = Counts()
	i = 0
	input = open(options["data"] + training_file, "rU")
	for s in input:
		i += 1
		if i %1000 == 0:
			print i
		counter.nltk_count(s)

	# Test: use to parsed sentences
	print "Testing..."

	with CYK(PCFG(counter)) as parser:
		output = open(options["results"] + "/cyk_bnc_init.results.txt", "w+")
		for line in open(options["data"] + testing_file):
			print line
			result = parser.parse(line, "TOP")
			if len(result) > 0:
				# result_str = JSONEncoder().encode(result)
				# print result_str
				# output.write(result_str + "\n")

				# pprint.pprint(result)
				resultTree = MyTree.from_list(result)
				print resultTree

				pos = " ".join([pos for word, pos in resultTree.pos()]) + "\n"
				output.write(pos)
				print "\n"+pos

				resultTree.un_chomsky_normal_form()
				print resultTree

		output.close()

if __name__ == "__main__":
	if len(sys.argv) < 2:
		sys.exit(1)
	main(sys.argv[1], sys.argv[2])
