class ArcHybridParse(object):
	""" Code taken from Matthew Honnibal's blog post: 
		https://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing/
		Design taken from: Goldberg; Nivre (2013) Training Deterministic Parsers with Non-Deterministic Oracles
	"""
	def __init__(self, n):
		self.n = n
		self.heads = [None] * (n-1)
		self.lefts = []
		self.rights = []
		for i in range(n+1):
			self.lefts.append(DefaultList(0))
			self.rights.append(DefaultList(0))
 
	def add_arc(self, head, child):
		self.heads[child] = head
		if child < head:
			self.lefts[head].append(child)
		else:
			self.rights[head].append(child)
	
	# Each step of the parsing process applies one of three actions to the state
	SHIFT = 0; RIGHT = 1; LEFT = 2
	MOVES = [SHIFT, RIGHT, LEFT]
 
	def transition(move, i, stack, parse):
		global SHIFT, RIGHT, LEFT
		if move == SHIFT:
			stack.append(i)
			return i + 1
		elif move == RIGHT:
			parse.add_arc(stack[-2], stack.pop())
			return i
		elif move == LEFT:
			parse.add_arc(i, stack.pop())
			return i
		raise GrammarError("Unknown move: %d" % move)

	def parse(self, words):
		tags = self.tagger(words)
		n = len(words)
		idx = 1
		stack = [0]
		deps = Parse(n)
		while stack or idx < n:
			features = extract_features(words, tags, idx, n, stack, deps)
			scores = self.model.score(features)
			valid_moves = get_valid_moves(i, n, len(stack))
			next_move = max(valid_moves, key=lambda move: scores[move])
			idx = transition(next_move, idx, stack, parse)
		return tags, parse
 
	def get_valid_moves(i, n, stack_depth):
		moves = []
		if i < n:
			moves.append(SHIFT)
		if stack_depth >= 2:
			moves.append(RIGHT)
		if stack_depth >= 1:
			moves.append(LEFT)
		return moves

 
	def extract_features(words, tags, n0, n, stack, parse):
		def get_stack_context(depth, stack, data):
			if depth >;= 3:
				return data[stack[-1]], data[stack[-2]], data[stack[-3]]
			elif depth >= 2:
				return data[stack[-1]], data[stack[-2]], ''
			elif depth == 1:
				return data[stack[-1]], '', ''
			else:
				return '', '', ''
 
		def get_buffer_context(i, n, data):
			if i + 1 >= n:
				return data[i], '', ''
			elif i + 2 >= n:
				return data[i], data[i + 1], ''
			else:
				return data[i], data[i + 1], data[i + 2]
 
		def get_parse_context(word, deps, data):
			if word == -1:
				return 0, '', ''
			deps = deps[word]
			valency = len(deps)
			if not valency:
				return 0, '', ''
			elif valency == 1:
				return 1, data[deps[-1]], ''
			else:
				return valency, data[deps[-1]], data[deps[-2]]

		features = {}
		# Set up the context pieces --- the word, W, and tag, T, of:
		# S0-2: Top three words on the stack
		# N0-2: First three words of the buffer
		# n0b1, n0b2: Two leftmost children of the first word of the buffer
		# s0b1, s0b2: Two leftmost children of the top word of the stack
		# s0f1, s0f2: Two rightmost children of the top word of the stack
 
		depth = len(stack)
		s0 = stack[-1] if depth else -1
 
		Ws0, Ws1, Ws2 = get_stack_context(depth, stack, words)
		Ts0, Ts1, Ts2 = get_stack_context(depth, stack, tags)
 
		Wn0, Wn1, Wn2 = get_buffer_context(n0, n, words)
		Tn0, Tn1, Tn2 = get_buffer_context(n0, n, tags)
 
		Vn0b, Wn0b1, Wn0b2 = get_parse_context(n0, parse.lefts, words)
		Vn0b, Tn0b1, Tn0b2 = get_parse_context(n0, parse.lefts, tags)
 
		Vn0f, Wn0f1, Wn0f2 = get_parse_context(n0, parse.rights, words)
		_, Tn0f1, Tn0f2 = get_parse_context(n0, parse.rights, tags)
 
		Vs0b, Ws0b1, Ws0b2 = get_parse_context(s0, parse.lefts, words)
		_, Ts0b1, Ts0b2 = get_parse_context(s0, parse.lefts, tags)
 
		Vs0f, Ws0f1, Ws0f2 = get_parse_context(s0, parse.rights, words)
		_, Ts0f1, Ts0f2 = get_parse_context(s0, parse.rights, tags)
 
		# Cap numeric features at 5? 
		# String-distance
		Ds0n0 = min((n0 - s0, 5)) if s0 != 0 else 0
 
		features['bias'] = 1
		# Add word and tag unigrams
		for w in (Wn0, Wn1, Wn2, Ws0, Ws1, Ws2, Wn0b1, Wn0b2, Ws0b1, Ws0b2, Ws0f1, Ws0f2):
			if w:
				features['w=%s' % w] = 1
		for t in (Tn0, Tn1, Tn2, Ts0, Ts1, Ts2, Tn0b1, Tn0b2, Ts0b1, Ts0b2, Ts0f1, Ts0f2):
			if t:
				features['t=%s' % t] = 1
 
		# Add word/tag pairs
		for i, (w, t) in enumerate(((Wn0, Tn0), (Wn1, Tn1), (Wn2, Tn2), (Ws0, Ts0))):
			if w or t:
				features['%d w=%s, t=%s' % (i, w, t)] = 1
 
		# Add some bigrams
		features['s0w=%s,  n0w=%s' % (Ws0, Wn0)] = 1
		features['wn0tn0-ws0 %s/%s %s' % (Wn0, Tn0, Ws0)] = 1
		features['wn0tn0-ts0 %s/%s %s' % (Wn0, Tn0, Ts0)] = 1
		features['ws0ts0-wn0 %s/%s %s' % (Ws0, Ts0, Wn0)] = 1
		features['ws0-ts0 tn0 %s/%s %s' % (Ws0, Ts0, Tn0)] = 1
		features['wt-wt %s/%s %s/%s' % (Ws0, Ts0, Wn0, Tn0)] = 1
		features['tt s0=%s n0=%s' % (Ts0, Tn0)] = 1
		features['tt n0=%s n1=%s' % (Tn0, Tn1)] = 1
 
		# Add some tag trigrams
		trigrams = ((Tn0, Tn1, Tn2), (Ts0, Tn0, Tn1), (Ts0, Ts1, Tn0), 
					(Ts0, Ts0f1, Tn0), (Ts0, Ts0f1, Tn0), (Ts0, Tn0, Tn0b1),
					(Ts0, Ts0b1, Ts0b2), (Ts0, Ts0f1, Ts0f2), (Tn0, Tn0b1, Tn0b2),
					(Ts0, Ts1, Ts1))
		for i, (t1, t2, t3) in enumerate(trigrams):
			if t1 or t2 or t3:
				features['ttt-%d %s %s %s' % (i, t1, t2, t3)] = 1
 
		# Add some valency and distance features
		vw = ((Ws0, Vs0f), (Ws0, Vs0b), (Wn0, Vn0b))
		vt = ((Ts0, Vs0f), (Ts0, Vs0b), (Tn0, Vn0b))
		d = ((Ws0, Ds0n0), (Wn0, Ds0n0), (Ts0, Ds0n0), (Tn0, Ds0n0),
			('t' + Tn0+Ts0, Ds0n0), ('w' + Wn0+Ws0, Ds0n0))
		for i, (w_t, v_d) in enumerate(vw + vt + d):
			if w_t or v_d:
				features['val/d-%d %s %d' % (i, w_t, v_d)] = 1
		return features
		
	def train_one(self, itn, words, gold_tags, gold_heads):
		n = len(words)
		i = 2; stack = [1]; parse = Parse(n)
		tags = self.tagger.tag(words)
		while stack or (i + 1) < n:
			features = extract_features(words, tags, i, n, stack, parse)
			scores = self.model.score(features)
			valid_moves = get_valid_moves(i, n, len(stack))
			guess = max(valid_moves, key=lambda move: scores[move])
			gold_moves = get_gold_moves(i, n, stack, parse.heads, gold_heads)
			best = max(gold_moves, key=lambda move: scores[move])
		self.model.update(best, guess, features)
		i = transition(guess, i, stack, parse)
	# Return number correct
	return len([i for i in range(n-1) if parse.heads[i] == gold_heads[i]])
	
	def get_gold_moves(n0, n, stack, heads, gold):
		def deps_between(target, others, gold):
			for word in others:
				if gold[word] == target or gold[target] == word:
					return True
			return False
 
		valid = get_valid_moves(n0, n, len(stack))
		if not stack or (SHIFT in valid and gold[n0] == stack[-1]):
			return [SHIFT]
		if gold[stack[-1]] == n0:
			return [LEFT]
		costly = set([m for m in MOVES if m not in valid])
		# If the word behind s0 is its gold head, Left is incorrect
		if len(stack) >= 2 and gold[stack[-1]] == stack[-2]:
			costly.add(LEFT)
		# If there are any dependencies between n0 and the stack,
		# pushing n0 will lose them.
		if SHIFT not in costly and deps_between(n0, stack, gold):
			costly.add(SHIFT)
		# If there are any dependencies between s0 and the buffer, popping
		# s0 will lose them.
		if deps_between(stack[-1], range(n0+1, n-1), gold):
			costly.add(LEFT)
			costly.add(RIGHT)
		return [m for m in MOVES if m not in costly]