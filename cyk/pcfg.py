from memorise import memoised


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
			:return: MLE of given rule.
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
				print "Using pseudo word: ", pseudo_word, " for ", R
				return float(self.counts.unary[(X, pseudo_word)]) / self.counts.nonterm[X]
		return 0

	def get_unary_rules_for(self, rhs_list):
		"""
		Lookup unary rules given the RHS rule
		:param rhs_list: Ys if the rule is X -> Y
		:return:
		"""
		out = set()
		for Y in rhs_list:
			if Y in self.counts.reverse_unary.keys():
				for X in self.counts.reverse_unary[Y]:
					print "Adding unary ", (X, Y)
					out.add((X, Y))
		return out

	def lookup_rules_for(self, rhs_left_corners, rhs_right_corners):
		""" Look up the rule heads based on RHS of rules.
			i.e. X -> Y Z filter rules by Y then Z
			:param rhs_left_corners: list of RHS left corner POS tags
			:param rhs_right_corners: list of RHS right corner POS tags
			:return: list of binary rule tuples
		"""
		rules = set()
		for Y, count in rhs_left_corners.iteritems():
			if count > 0:
				rule = self.counts.get_binary_by_left_corner(Y)
				if not rule is None:
					assert len(rule) == 3
					rules.add(rule)
		out = set()
		for (X, Y, Z) in rules:
			if Z in rhs_right_corners:
				out.add((X, Y, Z))
		return out

	def get_word_pos_tags(self, word):
		return self.counts.get_pos_tags(word)