#! /usr/bin/python

__author__="Alexander Whillas <whillas@gmail.com>"
__date__ ="$Nov 6, 2014"

#from Counts import Counts

class PCFG:
	
	def __init__(self, counts, pseudo_word_freq = 5):
		""" counts: Counts object.
			smoothing: type of smoothing. only 'Laplace Add-1' at the moment.
		"""
		self.counts = counts
		self.counts.mapToPseudoWords(pseudo_word_freq)
		
	def P(self, R):
		""" Maximum-likelihood estimate (MLE) of the given 'X -> y Z' or 'X -> word'
			production rule.
			R: Tuple of the form (X, Y, Z) or (X, word)
			return: MLE of given rule.
		"""
		X = R[0]
		if len(R) > 2:
			# Binary rule
			if R in self.counts.binary.keys():
				return float(self.counts.binary[R]) / self.counts.nonterm[X]
		else:
			# unarys rule
			pseudo_word = self.counts.pseudo_map(R[1])
			if R in self.counts.unary.keys():
				return float(self.counts.unary[R]) / self.counts.nonterm[X]
			elif (X, pseudo_word) in self.counts.unary.keys():   # map to pseudo word
				return float(self.counts.unary[(X, pseudo_word)]) / self.counts.nonterm[X]
		return 0
	
	def unarys(self):
		return self.counts.T.keys()
	
	def isUnary(self, A):
		return A in self.counts.unary.keys()
		
	def binarys(self):
		return self.counts.binary.keys()
	
	def isBinaryRule(self, N):
		""" Get all the production rules for the non-terminal N
		"""
		return N in self.counts.N and len(self.counts.N[N]) > 1

	def isNonTerminal(self, X):
		return X in self.counts.unary.keys() or X in self.counts.nonterm.keys()

	def getBinaryRulesFor(self, X):
		if X in self.counts.N.keys():
			print "Rule: ", X, " has ", len(self.counts.N[X]), " rules"
			return self.counts.N[X]
		else:
			print "NO binary rule for ", X
			return {}

	def getUnaryRulesFor(self, rhs_list):
		"""
		Lookup unary rules given the RHS rule
		:param rhs_list: Ys if the rule is X -> Y
		:return:
		"""
		out = set()
		for Y in rhs_list:
			if Y in self.counts.reverseT.keys():
				for X in self.counts.reverseT[Y]:
					out.add((X, Y))
		return out

	def lookupRulesFor(self, rhs_pair):
		""" Look up the rule heads based on RHS of rules.
			:param rhs_pair: list of RHS of binary rule tuples
			:return: list of binary rule tuples
		"""
		out = set()
		for Y, Z in rhs_pair:
			lhs_pairs = self.counts.reverseLookup((Y, Z))
			for X in lhs_pairs:
				out.add((X, Y, Z))
		return out

	def getWordTags(self, word):
		if word in self.counts.word_tags.keys():
			return self.counts.word_tags[word]
		else:
			# could be any unary tag so return them all
			return self.counts.T.keys()