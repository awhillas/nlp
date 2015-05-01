__author__ = 'alex'

import re
import string
from math import exp, log, fsum
from collections import defaultdict
from sortedcontainers import SortedDict  # see http://www.grantjenks.com/docs/sortedcontainers/sorteddict.html
from scipy.optimize import minimize
from itertools import izip, izip_longest
from scipy import array
import time

# Common functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def normalize(d, target=1.0):
	""" Make all the values add to target
	:param d: dict
	:param target: target sum, defaults to 1.0
	:return: dict
	"""
	raw = fsum(d.itervalues())
	factor = target / raw
	return {key: value * factor for key, value in d.iteritems()}


# Interfaces
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SequenceModel(object):
	""" Model for machine learning models
	"""

	def __init__(self):
		pass

	def train(self, data):
		""" Train the model from a corpus.
		:param data: Expect an object with an interface of CorpusReader from the NLTK
		:return: A model that can be passed to the label method
		"""
		raise NotImplementedError("Should have implemented this")

	def label(self, sequences):
		""" Predict labels for the list of sentences
		:param sequences: list of sentences which are lists of words.
		:return: labeled sentences which are lists of (word, label) tuples
		"""
		raise NotImplementedError("Should have implemented this")

	def get_labels(self, word):
		""" Lookup the tags for a word i.e. what Ratnaparkhi called a Tag Dictionary.
			For unseen words this should return all tags.
		:param word: The word to find tags for
		:return: list of tags
		"""
		raise NotImplementedError("Should have implemented this")

	def potential(self, state, previous_state, context, i):
		""" Returns the potential/likelihood? of the given state given the previous state at a position in the context
		:param state: hypothesised state
		:param previous_state: hypothesised previous state
		:param context: list of (word, label) pairs. Labels should be None where not known.
		:param i: index into the context
		:return: probability/likelihood of the state
		"""
		raise NotImplementedError("Should have implemented this")


class SequenceFeaturesTemplate(object):
	""" Interface to features used in sequencing models like MaxEnt or Log Linear etc
	"""

	# How many words passed the current word (i) do the feature templates look. Context object uses this.
	LOOK_BEHIND = 2  # passed the beginning
	LOOK_AHEAD = 2  # after the last word

	def __init__(self):
		pass

	@classmethod
	def get(cls, i, context):
		"""
		:param i: index into the sequence that we are up to.
		:param context: sequence context we are labeling i.e. a words+tags in POS tagging for example.
		:return: list of features present for the i'th position in the sequence, suitable for a hash key.
		"""
		raise NotImplementedError("Should have implemented this")

	@classmethod
	def get_suffixes(cls, word, n):
		""" Get n suffixes from a word. Word should be longer than n+1 """
		if len(word) <= 1:
			return []
		if not len(word) > n:
			n = len(word)

		return [word[-i:] for i in range(1, n)]


class WordNormaliser(object):
	"""
	Handles normalisation of words. Used to close vocabulary.
	"""

	def __init__(self):
		pass

	@classmethod
	def all(cls, data):
		out = []
		for sequence in data:
			words, labels = zip(*sequence)
			wordz = cls.sentence(words)
			out.append(zip(*(wordz, labels)))
		return out

	@classmethod
	def sentence(cls, sentence):
		""" Normalise a sentence
		:param sentence: List of words
		:return: List of normaised words
		"""
		return [cls.word(word) for word in sentence]

	@classmethod
	def word(cls, word):
		""" Normalise a word.
		:param word: A word
		:return: Normalised word.
		"""
		raise NotImplementedError("Should have implemented this")


# Implementations
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# class LogLinearModel(SequenceModel):
# 	""" aka Maximum Entropy Model
# 		TODO: implement for comparison purposes.
# 	"""
# 	def __init__(self):
# 		super(LogLinearModel, self).__init__()


class MaxEntMarkovModel(SequenceModel):
	""" Maximum Entropy Markov Model
		What makes it Markov is that the previous two labels are part of the
		features which are passed on via th context.
	"""

	def __init__(self, data, feature_templates, word_normaliser, regularization_parameter=0.5):
		"""
		:param feature_templates: Instance of SequenceFeaturesTemplate
		:param word_normaliser: Instance of WordNormaliser
		:param regularization_parameter: Parameter used to tune the regularization amount.
		:return:
		"""
		super(MaxEntMarkovModel, self).__init__()
		self.feature_templates = feature_templates
		self.normaliser = word_normaliser
		self.normalised_data = self.normaliser.all(data)
		self.regularization_parameter = regularization_parameter
		self.learnt_features = SortedDict()  # all features broken down into counts for each label
		self.learnt_features_full = SortedDict()  # full features including labels
		self.parameters = SortedDict()  # lambdas aka weights aka model parameters
		self.total = 0  # Total words seen in training corpus
		self.tag_count = {}  # Keep a count of each tag
		self.word_tag_count = {}  # Keep track of word -> tag -> count

	def get_labels(self, word):
		if word in self.word_tag_count:
			return self.word_tag_count[word].keys()
		else:
			return self.tag_count.keys()

	def train(self, data=None):
		"""
		:param data: List of sentences, Sentences are lists if (word, label) sequences.
		:return: true on success, false otherwise
		"""
		if data is None:
			data = self.normalised_data

		for sentence in data:
			context = Context(sentence, self.feature_templates)
			for i, word in enumerate(context.words):
				label = context.labels[i]
				self.add_tag(word, label)
				for f in context.features[i]:
					self.learnt_features.setdefault(f, {})
					self.learnt_features[f].setdefault(label, 0)
					self.learnt_features[f][label] += 1  # Keep counts of features by tag for gradient.
				for f in context.get_features(i, label):
					self.learnt_features_full.setdefault(f, 0)
					self.learnt_features_full[f] += 1
					self.parameters.setdefault(f, 1.0)  # initial default to 1.0

	def learn_parameters(self, data=None, maxiter=20):
		""" Learn the parameter vector (weights) for the model
		:param data: List of sentences, Sentences are lists if (word, label) sequences.
		:return: True on success, False otherwise
		"""
		print "Estimating parameters"

		if data is None:
			data = self.normalised_data

		def objective(x):  # Objective function
			v = SortedDict(izip(self.parameters.iterkeys(), x))
			log_p = 0.0
			for seq in data:
				for i, (word, label) in enumerate(seq):
					probabilities = self.probabilities(i, Context(seq, self.feature_templates), v)
					log_p += log(probabilities[label])

			# Regularization
			regulatiser = sum([param * param for param in v.itervalues()]) * (self.regularization_parameter / 2)

			print log_p, regulatiser, log_p - regulatiser
			return log_p - regulatiser

		def inverse_gradient(x):
			""" Inverse (coz we want the max. not min.) of the Gradient of the objective
			"""
			v = SortedDict(izip(self.parameters.iterkeys(), x))  # current param. vector
			dV = SortedDict.fromkeys(self.parameters.iterkeys(), 0.0)  # gradient vector output

			# Expected/predicted feature counts

			for n, seq in enumerate(data):
				# print "#", n, (float(n) + 1) / len(data) * 100, "%"
				context = Context(seq, self.feature_templates)

				for i, _ in enumerate(seq):
					probabilities = self.probabilities(i, context, v)

					for label in self.tag_count.iterkeys():
						for feature in context.get_features(i, label):
							if feature in v:
								# print "dV[", feature, "] += ", probabilities[label]
								dV[feature] += probabilities[label]

			# Actual feature counts + regularize

			for f, count in self.learnt_features_full.iteritems():
				dV[f] -= count
				dV[f] += v[f] * self.regularization_parameter

			return array(dV.values())

		# Maximise, actually.
		result = minimize(fun=lambda x: -objective(x), jac=lambda x: inverse_gradient(x), x0=self.parameters.values(), method='L-BFGS-B', options={'maxiter': maxiter})

		self.parameters = SortedDict(izip(self.learnt_features_full.iterkeys(), result.x.tolist()))
		if not result.success:
			print result.message
		return True

	def label(self, unlabeled_sequence):
		""" Prediction: Calls the Viterbi algorithm to label each sentence in the input sequence
		:param unlabeled_sequence: List of sentences, which are lists of words.
		:return: List if labeled sentences: lists of (word, tag) tuple pairs.
		"""

		def print_sol(words, predicted):
			row_format = ''
			for j, w in enumerate(words):
				row_format += "{"+str(j+1)+":<"+str(len(w)+2)+"}"
			print row_format.format("words: ", *words)
			print row_format.format("tagged:", *predicted)

		out = []
		all_tags = self.tag_count.keys()
		for i, raw_seq in enumerate(unlabeled_sequence, start=1):
			print "\nSentence {0} ({1:2.2f}%)".format(i, float(i)/len(unlabeled_sequence) * 100)
			seq = self.normaliser.sentence(raw_seq)
			t0 = time.time()
			context = Context((seq, [''] * len(seq)), self.feature_templates)
			start_p = dict([(t, self.potential(t, Context.BEGIN_SYMBOL, context, 0)) for t in all_tags])
			prob, tags = Viterbi.viterbi(seq, all_tags, start_p, self)
			t1 = time.time()
			print_sol(seq, tags)
			print "Time:", '%.3f' % (t1 - t0), ", Per word:", '%.3f' % ((t1 - t0) / len(seq))
			out.append(zip(raw_seq, tags))
		return out

	def potential(self, label, prev_label, context, i):
		context.labels[i] = label
		if len(context.sequence) > 1:
			context.labels[i-1] = prev_label
		probs = self.probabilities(i, context, self.parameters)
		return probs[label]

	def probabilities(self, i, context, v):
		""" Gets the probability distribution for all the tags/classes/labels for the current word/item in the
			sentence/sequence with a given feature.
			aka softmax function
		:param i: int. position in the context
		:param context: Context object.
		:param v: dict. parameter weight vector for each feature
		:return: dict. of class->float
		"""
		class_probabilities = dict()
		for label in self.tag_count.iterkeys():
			class_probabilities.setdefault(label, 1.0)  # coz exp(0) = 1
			for feature in context.get_features(i, label):
				if feature in v:
					class_probabilities[label] *= exp(v[feature])  # adding exponents is the same as * them
		return normalize(class_probabilities)

	def add_tag(self, w, t):
		""" Keep track of words and their tags
		:param w: word
		:param t: tag/label
		"""
		self.total += 1
		self.tag_count.setdefault(t, 0)
		self.tag_count[t] += 1
		self.word_tag_count.setdefault(w, defaultdict(int))
		self.word_tag_count[w][t] += 1


class CollinsNormalisation(WordNormaliser):
	""" Normalisations taken from Michel Collins notes
		'Chapter 2: Tagging problems and Hidden Markov models'
	"""

	@classmethod
	def word(cls, word):
		# return cls.pseudo_map_digits(word)
		out = cls.pseudo_map_digits(word)
		if out == word:
			return cls.junk(word)
		else:
			return out

	@classmethod
	def junk(cls, word):
		if len(word) > 1 \
				and "'" not in word \
				and '-' not in word \
				and not word[-1] == '.':
			if word.count('@') == 1:  # poor mans email spotter
				return '!emailAddress'
			elif word.lower().startswith('http') \
					or word.lower().startswith('www.') \
					or word.lower().endswith('.com'):  # poor mans URL
				return '!url'
			elif all(i in string.punctuation for i in word):
				# TODO: handle smiles
				return '!allPunctuation'
			elif not all(i in string.letters for i in word):
				# TODO: handle URLs
				# TODO: email addresses
				# TODO: handle initials
				# TODO: name titles i.e. Mr. Dr. etc. St. i.e e.g. T.V.
				# print '!mixedUp', word
				return '!mixedUp'
		return word.lower()

	@classmethod
	def pseudo_map_digits(cls, word):
		""" Map a word to a pseudo word with digits.
			Want to apply this to all input words.
			See Michael Collins' NLP Coursera notes, chap.2
			:rtype: str
		"""
		_digits = re.compile('\d')

		def contains_digits(d):
			return bool(_digits.search(d))

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
			else:
				return '!containsDigit'
		return word

	@classmethod
	def low_freq_words(cls, word):
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
			return '!initCap'  # TODO: should distinguish between words at beginning of sentence?

		if word.islower():
			return '!lowercase'

		return "!other"  # weird punctuation etc


class HonnibalFeats(SequenceFeaturesTemplate):
	""" Features nicked from Mathew Honnibal's PerceptronTagger for TextBlob
	"""
	LOOK_BEHIND = 2  # passed the beginning
	LOOK_AHEAD = 2  # after the last word

	@classmethod
	def get(cls, i, context):
		"""Map tokens into a feature representation, implemented as a
		{hashable: float} dict. If the features change, a new model must be
		trained.
		:param i: position in the context
		:param context: tuple of (words, labels)
		"""

		def add(name, *args):
			features.append(' '.join((name,) + tuple(args)))

		def add_suffixes(feature, word, n):
			for s in cls.get_suffixes(word, n):
				add(feature, s)

		features = []  # Should be a set but that's too slow :(
		words, tags = context
		n_suffixes = 4
		# print i, words, tags

		add('bias')  # It's useful to have a constant feature, which acts sort of like a prior
		add('i word', words[i])
		add_suffixes('i suffix', words[i], n_suffixes)
		add('i pref1', words[i][0])

		add('i-1 word', words[i - 1])
		if i > HonnibalFeats.LOOK_BEHIND:
			add_suffixes('i-1 suffix', words[i-1], n_suffixes)
		if tags[i - 1] is not None:
			add('i-1 tag', tags[i-1])
		add('i-1 tag & i word', tags[i - 1], words[i])

		add('i-2 word', words[i - 2])
		if tags[i - 2] is not None:
			add('i-2 tag', tags[i - 2])
		if tags[i - 2] is not None and tags[i - 1] is not None:
			add('i tag & i-2 tag', tags[i - 1], tags[i - 2])

		add('i+1 word', words[i + 1])
		if i < len(words) - HonnibalFeats.LOOK_AHEAD - 1:
			add_suffixes('i+1 suffix', words[i+1], n_suffixes)
		add('i+2 word', words[i + 2])

		return features


class Context(object):
	"""
	The main idea of this class is to reduce the number of times feature_templates.get() and zip(*sequence) are called.
	"""
	# Symbol added to the beginning and end of the sequence so we don't fall of the edges """
	BEGIN_SYMBOL = '*'
	END_SYMBOL = '$'

	def __init__(self, sequence, feature_templates=HonnibalFeats):
		"""
		:param sequence: List of tuples of the form (word, label) or a tuble of lists (words, labels)
		:param feature_templates: A SequenceFeaturesTemplate object
		:return:
		"""
		extra = feature_templates.LOOK_BEHIND
		BEGIN = [Context.BEGIN_SYMBOL * i for i in range(1, extra+1)]
		END = [Context.END_SYMBOL * i for i in range(1, feature_templates.LOOK_AHEAD+1)]
		if isinstance(sequence, tuple):
			self.words, self.labels = sequence
			self.sequence = zip(*sequence)
		elif isinstance(sequence, list):
			self.sequence = sequence
			self.words, self.labels = zip(*sequence)
		else:
			print "Passing something weird to Context constructor:", type(sequence), sequence
			raise
		self.words = list(self.words)
		self.labels = list(self.labels)
		self.templates = feature_templates
		self.features = [[f for f in feature_templates.get(i+extra, (BEGIN+list(self.words)+END, BEGIN+list(self.labels)+END))] for i, _ in enumerate(self.sequence)]

	def get_features(self, i, label):
		return [f + " " + label for f in self.features[i]]  # merge the feature set with the label


class ForwardBackward(object):

	def __init__(self, model):
		self.model = model  # needs to implement interface with potential(state_from, state_to, context_index)

	def p(self, state_from, state_to, context, i):
		return self.model.potential(state_to, state_from, context, i)

	def forward_backward(self, seq, states):
		return self.fwd_bkw(seq, states, [1.0] * len(states), [1.0] * len(states))

	def fwd_bkw(self, seq, states, start, end):
		""" Iterative version of the Forward-Backwards algorithm
			Taken from Wikipedia
		:param seq: input (observation) sequence (sentence)
		:param states: states
		:param start: start probability
		:param end: end state
		:return: forward and backward probabilities + posteriors
		"""
		m = len(seq)

		# Forward part of the algorithm

		fwd = []
		f_prev = {}
		for i, x_i in enumerate(seq):
			f_curr = {}
			for st in states:
				if i == 0:
					# base case for the forward part
					prev_f_sum = start[st]
				else:
					prev_f_sum = sum(f_prev[k] * self.p(k, st, i) for k in states)

			f_curr = normalize(f_curr)

			# iterate (instead of recurse)
			fwd.append(f_curr)
			f_prev = f_curr

		z = sum(f_curr[k] for k in states)  # normalizer for the posteriors

		# Backward part of the algorithm

		bkw = []
		b_prev = {}
		for i, x_i_plus in enumerate(reversed(seq[1:] + (None,))):
			b_curr = {}
			for st in states:
				if i == 0:
					# base case for backward part
					b_curr[st] = end[st]
				else:
					b_curr[st] = sum(self.p(st, l, i) * b_prev[l] for l in states)

			b_curr = normalize(b_curr)

			# iterate
			bkw.insert(0, b_curr)
			b_prev = b_curr

		# p_bkw = sum(start[l] * e[l][seq[0]] * b_curr[l] for l in states) # Transition to first state?

		# merging the two parts
		posterior = []
		for i in range(m):
			posterior.append({st: fwd[i][st] * bkw[i][st] / z for st in states})

		#assert p_fwd == p_bkw
		return fwd, bkw, posterior


class Viterbi(object):
	@classmethod
	def viterbi(cls, seq, all_states, start_p, model):
		""" The Viterbi algorithm.
			assume that the observation sequence obs is non-empty and that
			trans_p[i][j] and emit_p[i][j] is defined for all states i,j.
			Taken from https://en.wikipedia.org/wiki/Viterbi_algorithm
		:param seq: the sequence of observations/items/words
		:param all_states: the set of states that make up the output sequence we are trying to produce
		:param start_p: start probability
		:param model: Trained model object that supports the interface:
			model.p(state, previous_state, context_obj, context_index) and
			model.get_labels(word)
		:return:
		"""
		V = [{}]  # len(seq) x len(states) dynamic programing table
		path = {}  # back pointers

		# Initialize base cases (t == 0)

		for s in all_states:
			V[0][s] = start_p[s]
			path[s] = [s]

		# Run Viterbi for j > 0

		for j in range(1, len(seq)):
			V.append(dict.fromkeys(all_states, 0))
			new_path = {}
			x = seq[j]  # current word
			for s in all_states:  # TODO: only consider labels we have seen for this word?
				context = Context(list(izip_longest(seq, path[s], fillvalue='')))
				(prob, state) = max( (V[j - 1][s0] * model.potential(s, s0, context, j), s0) for s0 in model.get_labels(seq[j-1]) )
				V[j][s] = prob
				new_path[s] = path[state] + [s]
			# Don't need to remember the old paths
			path = new_path

		# Find the max sequence

		n = 0  # if only one element is observed max is sought in the initialization values
		if len(seq) != 1:
			n = j
		#print_dptable(V, seq)
		(prob, state) = max((V[n][y], y) for y in all_states)
		return prob, path[state]


def print_dptable(V, seq):
	# Header
	row_format = "{:<10}  " * (len(seq) + 1)
	print row_format.format("", *seq)
	# Rows
	row_format ="{:>10}  " +  "{:-<1.8f}  " * (len(seq))
	inv = dict([(tag, [row[tag] for row in V]) for tag in V[0].keys()])  # transpose
	for tag, values in inv.iteritems():
		row = [tag] + values
		print row_format.format(*row)

class color:
	""" http://stackoverflow.com/questions/8924173/how-do-i-print-bold-text-in-python
		usage: print color.BOLD + 'Hello World !' + color.END
	"""
	PURPLE = '\033[95m'
	CYAN = '\033[96m'
	DARKCYAN = '\033[36m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	END = '\033[0m'
