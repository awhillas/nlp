#! /usr/bin/python

__author__ = "Alexander Whillas, Alexander Rush <srush@csail.mit.edu>"
__date__ = "$Sep 12, 2012"

import sys
import re
from nltk.tree import Tree


_digits = re.compile('\d')
def contains_digits(d):
	return bool(_digits.search(d))


class Counts:
	"""
	Count rule frequencies in CFG.
	This was nicked from the Coursera course on NLP.
	"""
	def __init__(self):
		self.unary = {}  # X -> terminal + X -> Y (i.e. non-terminal unarys)
		self.binary = {}  # X -> Y Z
		self.nonterm = {}  # Non-terminal rule frequency (unused?)
		self.N = {}  # Non-terminal counter. Rule RHS indexed by rule LHS
		#self.T = {}  # Terminal counter (no non-terminals)
		self.reverseN = {}  # Binary rule reverse lookup.
		self.reverseN_keys = []  # cache for the keys to speed up lookup
		self.reverseN_left_hand_corner = {}  # i.e. for X -> Y Z then index by Y
		self.reverse_unary = {}  # Unary rule reverse lookup i.e. indexed by Y
		self.word_pos = {}  # track {word -> set(tags)}.

	def count_trees(self, file_path, freq_threshold=5):
		with open(file_path, "rU") as data:
			i = 0
			for sentence in data:
				i += 1
				if i % 100 == 0:
					print i
				self.cnf_count(sentence)
		self.map_to_pseudo_words(freq_threshold)

	def cnf_count(self, s):
		"""
		Convert the tree to Chomsky Normal Form (CNF) and then count the tree data.
		Cleans the given string a little.
		:param s: String tree in bracketed form
		"""
		tree = Tree.fromstring(s.decode('utf-8').replace(u"\u00A0", " "))
		tree.chomsky_normal_form()
		self.count(tree)

	def count(self, tree):
		symbol = tree.label()

		if not symbol:
			return self.count(tree[0])

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

			self.reverseN_left_hand_corner.setdefault(y1, set())
			self.reverseN_left_hand_corner[y1].add(key)

			self.count(tree[0])
			self.count(tree[1])

		elif len(tree) == 1:        # It is a unary rule.

			if isinstance(tree[0], unicode):
				# X -> terminal
				y1 = tree[0]
				# self.T.setdefault(symbol, {})
				# self.T[symbol].setdefault(y1, 0)
				# self.T[symbol][y1] += 1
				self.word_pos.setdefault(y1, set())
				self.word_pos[y1].add(symbol)
			else:
				# X -> Y unary rules
				self.count(tree[0])  # recurs
				y1 = tree[0].label()
				self.reverse_unary.setdefault(y1, set())
				self.reverse_unary[y1].add(symbol)

			key = (symbol, y1)

			self.unary.setdefault(key, 0)
			self.unary[key] += 1

	def is_terminal(self, word):
		return word in self.word_pos

	def is_unary_non_terminal(self, Y):
		return Y in self.reverse_unary

	def is_unary(self, key):
		return key in self.unary

	def get_pos_tags(self, word):
		""" Get Parts-of-Speech tags for the given word """
		if self.is_terminal(word) or not self.is_unary_non_terminal(word):

			if self.have_seen(word):
				return self.word_pos[word]

			pseudo_word = self.normalise(word)
			if self.have_seen(pseudo_word):
				return self.word_pos[pseudo_word]

			return self.nonterm.keys()  # TODO: come up with something better if never seen word or its normaisation.
		else:
			return []

	def get_unary_rules_for(self, rhs_list):
		""" Lookup unary rules given the right-hand-side rule
		:param rhs_list: Ys if the rule is X -> Y
		:return: set
		"""
		out = set()
		for Y in rhs_list:
			if Y in self.reverse_unary:
				for X in self.reverse_unary[Y]:
					out.add((X, Y))
		return out

	def have_seen(self, word):
		""" Have we seen this word/terminal before? """
		return self.is_terminal(word)

	def get_binary_by_left_corner(self, Y):
		if Y in self.reverseN_left_hand_corner:
			return self.reverseN_left_hand_corner[Y]
		else:
			return []

	def map_to_pseudo_words(self, freq_threshold=5):
		""" Map words with a frequency less than the freq_threshold to pseudo-word """
		normaised = {}
		for (X, word), count in self.unary.iteritems():
			if not self.is_unary_non_terminal(word) and count < freq_threshold:
				pseudo = self.normalise(word)
				normaised.setdefault((X, pseudo), 0)
				normaised[(X, pseudo)] += count
				self.word_pos.setdefault(pseudo, set())
				self.word_pos[pseudo].add(X)
			else:
				normaised[(X, word)] = count
		self.unary = normaised  # Can't change the dict we're iterating though

	@classmethod
	def normalise(cls, word):
		out = cls.pseudo_map_digits(word)
		if out == word:
			return cls.pseudo_map(word)
		else:
			return out

	@classmethod
	def pseudo_map_digits(cls, word):
		""" Map a word to a pseudo word with digits.
			Want to apply this to all input words.
			See Michael Collins' NLP Coursera notes, chap.2
			:rtype: str
		"""
		if contains_digits(word):
			if word.isdigit():
				if len(word) == 2:
					return '!twoDigitNum'
				elif len(word) == 4:
					return '!fourDigitNum'
				else:
					return '!otherNumber'
			elif '-' in word:
				return '!containsDigitAndDash'
			elif '/' in word:
				return '!containsDigitAndSlash'
			elif ',' in word:
				return '!containsDigitAndComma'
			elif '.' in word:
				return '!containsDigitAndPeriod'
			elif word.isalnum():
				return '!containsDigitAndAlpha'
		return word

	@classmethod
	def pseudo_map(cls, word):
		""" Map a word to a pseudo word.
			Want to apply this only to lexicon words with low frequency.
			See Michael Collins' NLP Coursera notes, chap.2
			:rtype: str
		"""
		if len(word) > 1 and word[0].isupper() and word[1] == '.':
			return '!capPeriod'  # Person's name initial i.e. "M."

		if word.isupper():
			return '!allCaps'

		if word.istitle():
			return '!initCap'   # TODO: should distinguish between words at beginning of sentence?

		if word.islower():
			return '!lowercase'

		return "!other"     # weird punctuation etc

	# def reverse_lookup(self, rhs):
	# 	""" Get the head (LHS) of the production rules given the rules)
	# 	"""
	# 	if rhs in self.reverseN_keys:
	# 		return self.reverseN[rhs]
	# 	else:
	# 		return {}

def main(parse_file):
	counter = Counts()
	#counter.read_json_file(parse_file)
	for s in open(parse_file, "rU"):
		counter.cnf_count(s)

def usage():
	sys.stderr.write("""
	Usage: python Counts.py [tree_file]
		Counts the frequencies of structural elements of a given (Penn) Treebank.\n""")
		
if __name__ == "__main__":
	if len(sys.argv) != 2:
		usage()
		sys.exit(1)
	main(sys.argv[1])
