import operator

from lib import fix_bad_unicode


class CYK:
	""" Implementation of the classic CYK parsing algorithm
	"""
	def __init__(self, pcfg):
		self.pcfg = pcfg
		self.f = open("cyk_trouble_sents.txt", 'w')

	def __enter__(self):
		return self

	def __exit__(self, a_type, a_value, a_traceback):
		""" Like a destructor. Need to use this class like:
		```
		with CYK() as parser:
			# use parser object and __exit__() will be auto. called.
		```
		"""
		self.f.close()

	def parse(self, sentence_string, start_pos):
		cleaned_sentence = self.clean_string(sentence_string)
		return self.cyk(self.chunk(cleaned_sentence), start_pos)

	def write_file(self, line):
		self.f.write(line.encode('utf-8')+"\n")

	@staticmethod
	def chunk(string):
		""" Just chunk on spaces for now. 
			TODO: something more sophisticated.
			return: array of word strings.
		"""
		return string.split()

	@staticmethod
	def clean_string(string):
		""" Handle any encoding problems
		"""
		return fix_bad_unicode(unicode(string, 'utf-8'))

	def cyk(self, raw_words, start_pos):
		""" Cocke-Kasami-Younger (CKY) Constituency Parsing
			Based on notes from Standford NLP Coursera online course, 2012.
			We do have unary rules as we are using a non-strict CNF grammar.
		"""
		words = self.normaise(raw_words)
		n = len(words)  # Number of words in the sentence.
		score = [[{} for _ in range(n + 1)] for _ in range(n + 1)]  # Constituent likelihoods
		back = [[{} for _ in range(n + 1)] for _ in range(n + 1)]  # Back-pointers
		for i in range(0, n):
			pos = self.pcfg.get_word_pos_tags(words[i])
			for A in pos:  # unary non-terminals
				score[i][i+1][A] = self.pcfg.P((A, words[i]))
				back[i][i+1][A] = (0, words[i])
			# handle unarys...
			added = True
			while added:
				added = False
				for A, B in self.pcfg.get_unary_rules_for(score[i][i+1]):
					if B in score[i][i+1] and score[i][i+1][B] > 0:
						prob = self.pcfg.P((A, B)) * score[i][i+1][B]
						if not A in score[i][i+1] or prob > score[i][i+1][A]:
							score[i][i+1].setdefault(A, 0)
							score[i][i+1][A] = prob
							back[i][i+1].setdefault(A, (0, B))
							back[i][i+1][A] = (0, B)
							added = True
		for span in range(2, n+1):
			for begin in range(0, n - span + 1):
				end = begin + span
				for split in range(begin+1, end):
					for (A, B, C) in self.pcfg.lookup_rules_for(score[begin][split], score[split][end]):
						score[begin][end].setdefault(A, 0)
						if B in score[begin][split] and C in score[split][end]:
							prob = score[begin][split][B] * score[split][end][C] * self.pcfg.P((A, B, C))
						if prob > score[begin][end][A]:
							score[begin][end][A] = prob
							back[begin][end][A] = (split, B, C)
				# handle unarys...
				added = True
				while added:
					added = False
					for A, B in self.pcfg.get_unary_rules_for(score[begin][end]):
						prob = self.pcfg.P((A, B)) * score[begin][end][B]
						if not A in score[begin][end] or prob > score[begin][end][A]:
							score[begin][end].setdefault(A, prob)
							score[begin][end][A] = prob
							back[begin][end][A] = (0, B)
							added = True

		# if len(back[0][len(back[0])-1]) == 0:
		# 	self.write_file(" ".join(words)) # keep problem sentences for debug
		return self.build_tree(score, back, start_pos)

	def normaise(self, words):
		out = []
		for w in words:
			out.append(self.pcfg.counts.pseudo_map_digits(w))
		return out

	def build_tree(self, score, bp, start_pos):
		""" Reconstruct the best parse tree from the give back pointers
		"""
		row = 0
		col = len(bp[0])-1
		if len(score[row][col]):
			if start_pos in score[row][col]:
				root = start_pos
			else:
				root = max(score[row][col].iteritems(), key=operator.itemgetter(1))[0]  # Find the max tree root
			return self.get_subtree(root, (row, col), score, bp)
		else:
			return []

	def get_subtree(self, constituent, position, score, bp):
		""" Rebuild the most probable parse from the score recursively
		:param constituent: most probably parent constituent
		:param position: where in the chart the constituent is
		:param score: table of probabilities built by the CYK algorithm
		:param bp: table of back-pointers of the splits and sub constituents
		:return: dict
		"""
		row = position[0]
		col = position[1]
		if len(bp[row][col][constituent]) == 3:
			split, left, right = bp[row][col][constituent]
			l = self.get_subtree(left, (row, split), score, bp)
			r = self.get_subtree(right, (split, col), score, bp)
			return [constituent, l, r]
		else:
			split, left = bp[row][col][constituent]
			if left != constituent and self.pcfg.is_non_terminal(left):
				return [constituent, self.get_subtree(left, position, score, bp)]
			else:
				return [constituent, left]

	@staticmethod
	def pretty(matrix):
		""" Prints a 2D table nice!
			:param matrix: 2D array to print pretty
		"""
		s = [[str(e) for e in row] for row in matrix]
		lens = [max(map(len, col)) for col in zip(*s)]
		fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
		table = [fmt.format(*row) for row in s]
		print '\n'.join(table)
