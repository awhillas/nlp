#! /usr/bin/python

__author__ = "Alexander Whillas, Alexander Rush <srush@csail.mit.edu>"
__date__ = "$Sep 12, 2012"

import sys
import json
from nltk.tree import Tree
#from nltk.corpus import BracketParseCorpusReader
#from nltk.corpus import treebank	# Expects the data in its corpus repo :(


class Counts:
	"""
	Count rule frequencies in a binarised CFG.
	This was nicked from the Coursera course on NLP.
	"""
	def __init__(self):
		self.unary = {}
		self.binary = {}
		self.nonterm = {}
		self.N = {}  # Non-terminal counter
		self.T = {}  # Terminal counter
		self.reverseN = {}  # Binary rule reverse lookup.
		self.reverseT = {}  # Unary rule reverse lookup.
		self.word_tags = {} # track word tags.

	def show(self, X):
		for Y, Z in self.N[X]:
			print X, '->', Y, Z

	def nltk_count(self, s):
		"""
		Count the tree data.
		:param s: String tree in bracketed form
		"""
		tree = Tree.fromstring(s.decode('utf-8').replace(u"\u00A0", " "))
		tree.chomsky_normal_form()
		self._nltk_count(tree)

	def _nltk_count(self, tree):
		symbol = tree.label()

		if not symbol:
			return self._nltk_count(tree[0])

		self.nonterm.setdefault(symbol, 0)
		self.nonterm[symbol] += 1

		if len(tree) == 2:          # It is a binary rule.

			y1 = tree[0].label()
			y2 = tree[1].label()
			key = (symbol, y1, y2)

			self.binary.setdefault(key, 0)
			self.binary[(symbol, y1, y2)] += 1

			self.N.setdefault(symbol, {})
			self.N[symbol].setdefault((y1, y2), 0)
			self.N[symbol][(y1, y2)] += 1

			self.reverseN.setdefault((y1, y2), set())
			self.reverseN[(y1, y2)].add(symbol)

			self._nltk_count(tree[0])
			self._nltk_count(tree[1])

		elif len(tree) == 1:        # It is a unary rule.

			if isinstance(tree[0], unicode):
				# unary -> terminal
				y1 = tree[0]
			else:
				# unary non-terminal -> non-terminal
				self._nltk_count(tree[0])
				y1 = tree[0].label()

			key = (symbol, y1)

			self.unary.setdefault(key, 0)
			self.unary[key] += 1

			self.T.setdefault(symbol, {})
			self.T[symbol].setdefault(y1, 0)
			self.T[symbol][y1] += 1

			self.reverseT.setdefault(y1, set())
			self.reverseT[y1].add(symbol)

			self.word_tags.setdefault(y1, set())
			self.word_tags[y1].add(symbol)

	def count(self, tree):
		"""
		Count the frequencies of non-terminals and rules in the tree.
		"""
		if isinstance(tree, basestring): return

		# Count the non-terminal symbol.
		symbol = tree[0]
		if not tree[0]: return  #ignore empties

		self.nonterm.setdefault(symbol, 0)
		self.nonterm[symbol] += 1

		if len(tree) == 3:
			# It is a binary rule.
			y1, y2 = (tree[1][0], tree[2][0])
			key = (symbol, y1, y2)
			self.binary.setdefault(key, 0)
			self.binary[(symbol, y1, y2)] += 1

			self.N.setdefault(symbol, {})
			self.N[symbol].setdefault((y1, y2), 0)
			self.N[symbol][(y1, y2)] += 1

			self.reverseN[(y1, y2)] = symbol

			# Recursively count the children.
			self.count(tree[1])
			self.count(tree[2])

		elif len(tree) == 2:
			# It is a unary rule.
			if isinstance(tree[1], list):
				y1 = tree[1][0]     # unary with a sub-unary
				self.count(tree[1]) # count the sub-unarys
			else:
				y1 = tree[1]
			key = (symbol, y1)

			self.unary.setdefault(key, 0)
			self.unary[key] += 1

			self.T.setdefault(symbol, {})
			self.T[symbol].setdefault(y1, 0)
			self.T[symbol][y1] += 1

			self.word_tags.setdefault(y1, set())
			self.word_tags[y1].add(symbol)

	def mapToPseudoWords(self, threshold):
		""" Map words with a frequency less than the freq_threshold to pseudo-word
		"""
		new_unary_counts = {}
		for (X, word), count in self.unary.iteritems():
			if count < threshold:
				pseudo = self.pseudo_map(word)
				new_unary_counts.setdefault((X, pseudo), 0)
				new_unary_counts[(X, pseudo)] += count
			else:
				new_unary_counts[(X, word)] = count
		self.unary = new_unary_counts

	def pseudo_map(self, word):
		""" Map a word to a pseudo word
			atm all words are mapped to the same word.
			TODO: make more fine-grained.
		"""
		return "_RARE_"

	def read_json_file(self, parse_file):
		for l in open(parse_file):
			t = json.loads(l)	# Parse JSON data.
			self.count(t)

	def reverseLookup(self, rhs):
		""" Get the head (LHS) of the production rules given the rules)
		"""
		if rhs in self.reverseN.keys():
			return self.reverseN[rhs]
		else:
			return {}

def main(parse_file):
	counter = Counts()
	#counter.read_json_file(parse_file)
	for s in open(parse_file, "rU"):
		counter.nltk_count(s)

def usage():
	sys.stderr.write("""
	Usage: python Counts.py [tree_file]
		Counts the frequencies of structural elements of a given (Penn) Treebank.\n""")
		
if __name__ == "__main__":
	if len(sys.argv) != 2:
		usage()
		sys.exit(1)
	main(sys.argv[1])
