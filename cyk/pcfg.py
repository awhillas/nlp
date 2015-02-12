#from memorise import memoised


class PCFG:

	def __init__(self, counts, pseudo_word_freq=5):
		""" counts: Counts object.
			smoothing: type of smoothing. only 'Laplace Add-1' at the moment.
		"""
		self.counts = counts
		self.counts.map_to_pseudo_words(pseudo_word_freq)

	def P(self, R):
		""" Maximum-likelihood estimate (MLE) of the given 'X -> y Z' or 'X -> word'
			production rule.
			:param R: Tuple of the form (X, Y, Z) or (X, word) where X, Y and Z are tags.
			:return: float, MLE of given rule.
		"""
		X = R[0]
		if len(R) > 2:
			# Binary rule
			if R in self.counts.binary.keys():
				return float(self.counts.binary[R]) / self.counts.nonterm[X]
		else:
			# unarys rule
			pseudo_word = self.counts.pseudo_map(R[1])
			if self.counts.is_unary(R):
				return float(self.counts.unary[R]) / self.counts.nonterm[X]
			elif self.counts.is_unary((X, pseudo_word)):   # map to pseudo word
				return float(self.counts.unary[(X, pseudo_word)]) / self.counts.nonterm[X]
		return 0.0

	def get_unary_rules_for(self, rhs_list):
		return self.counts.get_unary_rules_for(rhs_list)

	def lookup_rules_for(self, rhs_left_corners, rhs_right_corners):
		""" Look up the rule heads based on RHS of rules.
			i.e. X -> Y Z filter rules by Y then Z
			:param rhs_left_corners: list of RHS left corner POS tags
			:param rhs_right_corners: list of RHS right corner POS tags
			:return: list of binary rule tuples
		"""
		candidates = set()
		for Y, count in rhs_left_corners.iteritems():
			if count > 0:
				for rule in self.counts.get_binary_by_left_corner(Y):
					assert len(rule) == 3
					candidates.add(rule)
		out = set()
		for (X, Y, Z) in candidates:
			if Z in rhs_right_corners:
				out.add((X, Y, Z))
		return out

	def get_word_pos_tags(self, word):
		return self.counts.get_pos_tags(word)

	def is_non_terminal(self, word):
		return self.counts.is_unary_non_terminal(word)