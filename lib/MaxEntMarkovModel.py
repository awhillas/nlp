# -*- coding: utf-8 -*-
__author__ = "Alexander Whillas <whillas@gmail.com>"

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


def print_dptable(V, seq):
	""" Display Dynamic Programming table for Viterbi algo
	"""
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


def matrix_to_string(matrix):
	# Figure out the max width of each column
	widths = [0] * len(matrix[0])
	for col in range(len(matrix[0])):
		for row in matrix:
			if len(row[col]) > widths[col]:
				widths[col] = len(row[col])
	# Generate a row format string
	row_format = ' '.join(["{:"+str(l)+"}" for l in widths])  # align'd right
	output = []
	for row in matrix:
		output += [row_format.format(*row)]
	return "\n".join(output)


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

			print "{:>8.2f} - {:>8.2f} = {:>8.2f}".format(log_p, regulatiser, log_p - regulatiser)
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
		params = self.parameters.values()
		if len(params) > 0:
			result = minimize(fun=lambda x: -objective(x), jac=lambda x: inverse_gradient(x), x0=params,\
							  method='L-BFGS-B', options={'maxiter': maxiter})
		else:
			print "No parameters to optimise!?"
			return False

		self.parameters = SortedDict(izip(self.learnt_features_full.iterkeys(), result.x.tolist()))
		if not result.success:
			print result.message
		return True

	def for_all(self, sequence_list, tagger, output_file = None):
		""" Loops though a list of sequences and applies the given function to each to get the corresponding tags.
			Also handles printing output.
		:param sequence_list: List of unlabeled sequences
		:param f: function to generate tags for each item/word in a sequence
		:param output_file: print the results to a file if given
		:return:
		"""
		out = []
		for i, unlabeled_sequence in enumerate(sequence_list, start=1):
			print "Sentence {0} ({1:2.2f}%)".format(i, float(i)/len(sequence_list) * 100)

			t0 = time.time()
			normalised_seq = self.normaliser.sentence(unlabeled_sequence)
			tags = tagger(normalised_seq)
			t1 = time.time()

			print matrix_to_string([unlabeled_sequence, normalised_seq, tags])
			print "Time:", '%.3f' % (t1 - t0), ", Per word:", '%.3f' % ((t1 - t0) / len(unlabeled_sequence)), "\n"
			out += [zip(unlabeled_sequence, tags)]
		return out

	def frequency_tag(self, unlabeled_sequence):
		""" Label with the highest frequency word label (base line)
		:param unlabeled_sequence: List of sentences, which are lists of words.
		:return: List if labeled sentences: lists of (word, tag) tuple pairs.
		"""
		highest_freq_tag = max(self.tag_count.iterkeys(), key=(lambda key: self.tag_count[key]))
		tags = []
		for word in unlabeled_sequence:
			if word in self.word_tag_count:
				best = max(self.word_tag_count[word].iterkeys(), key=(lambda key: self.word_tag_count[word][key]))
				tags += [best]
			else:
				# unseen word TODO: calculate an Unseen Word distribution using the CV set?
				tags += [highest_freq_tag]
		return tags

	def label(self, seq):
		""" Prediction: Calls the Viterbi algorithm to label each sentence in the input sequence
		:param seq: Lists of words to tag
		:return: List of tags.
		"""
		all_tags = self.tag_count.keys()
		context = Context((seq, [''] * len(seq)), self.feature_templates)
		start_p = dict([(t, self.potential(t, None, context, 0)) for t in all_tags])
		prob, tags = Viterbi.viterbi(seq, all_tags, start_p, self)
		return tags

	def potential(self, label, prev_label, context, i):
		""" Wrapper interface to the probabilities method. Sets the current and previous labels in the context before
			calling the method. Used for prediction on the Viterbi and Forward-Backwards algorithms
		:param label: Current label/tag
		:param prev_label: previous label/tag
		:param context: Context object
		:param i: index in the context at which we are getting the probabilities for
		:return:
		"""
		tags = context.labels
		tags[i] = label
		if len(context.sequence) > 1 and i > 0:
			tags[i-1] = prev_label
		# Need to recreate a new Context with the new tags so the features get regenerated.
		probs = self.probabilities(i, Context((context.words, tags)), self.parameters)
		return probs[label]

	def tag_probability_distributions(self, unlabeled_sequence):
		fb = ForwardBackward(self)
		fwd, bkw, posterior = fb.forward_backward(unlabeled_sequence, self.tag_count.keys())
		print fwd
		print bkw
		print posterior

	def probabilities(self, i, context, v=None):
		""" Gets the probability distribution for all the tags/classes/labels for the current word/item in the
			sentence/sequence with a given feature.
			aka softmax function
		:param i: int. position in the context
		:param context: Context object.
		:param v: dict. parameter weight vector for each feature. Defaults to the current version in the model.
		:return: dict. of label->probability at the given index in the context
		"""
		if v is None:
			v = self.parameters

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

	# Top 20 Emoticons: http://www.datagenetics.com/blog/october52012/index.html
	# TODO: get top 100 Emoticons
	EMOS = [':)', ':D', ':(', ';)', ':-)', ':P', '=)', '(:', ';-)', ':/', 'XD', '=D', ':o', '=]', 'D:', ';D', ':]', ':-(', '=/', '=(']
	RE_DIGITS = re.compile('\d')
	# See: http://code.tutsplus.com/tutorials/8-regular-expressions-you-should-know--net-6149]
	# TODO: uses these RegExs
	URL_REGEX = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|^www\..*|.*\.(com|org|net|edu|gov|co){1}(\.[a-zA-Z]{2})?$")
	EMAIL_REGEX = re.compile("^([a-z0-9_\.-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})$")
	IP_ADDRESS_REGEX = re.compile("^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")

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
		if word in cls.EMOS:
			# print '!emoticon!', word
			return '!emoticon!'
		if len(word) > 1 \
				and "'" not in word \
				and '-' not in word:
			if all(i in string.punctuation for i in word):
				return '!allPunctuation!'
			elif not all(i in string.letters for i in word):
				# TODO: name titles i.e. Mr. Dr. etc. St. i.e e.g. T.V.
				acro = '.'.join(''.join(c for c in word if c not in '.'))
				if acro == word or acro+'.' == word:
					if len(word) > 2:
						# U.S.A or U.S.A.
						# print '!acronym!', word
						return '!acronym!'
					else:
						# print '!initial!', word
						return '!initial!'
				elif word.count('.') == 1 and word[-1] == '.':
					# print '!abbreviation', word
					return '!abbreviation!'
				else:
					# TODO: handle smiles
					# print '!mixedUp!', word
					return '!mixedUp!'
		return word.lower()

	@classmethod
	def pseudo_map_digits(cls, word):
		""" Map a word to a pseudo word with digits.
			Want to apply this to all input words.
			See Michael Collins' NLP Coursera notes, chap.2
			:rtype: str
		"""
		if bool(cls.EMAIL_REGEX.search(word.lower())):
			# print '!email!', word
			return '!email!'
		elif bool(cls.URL_REGEX.search(word)):
			print '!url!', word
			return '!url!'
		elif bool(cls.RE_DIGITS.search(word)):  # contains digits
			if word.isdigit():
				if len(word) == 2:
					return '!twoDigitNum!'
				elif len(word) == 4:
					return '!fourDigitNum!'
				else:
					return '!otherNum!'
			elif '-' in word:
				return '!containsDigitAndDash!'
			elif '/' in word:
				return '!containsDigitAndSlash!'
			elif ',' in word:
				return '!containsDigitAndComma!'
			elif '.' in word:
				return '!containsDigitAndPeriod!'
			elif word.isalnum():
				return '!containsDigitAndAlpha!'
			else:
				return '!containsDigit!'
		return word

	@classmethod
	def low_freq_words(cls, word):
		""" Map a word to a pseudo word.
			Want to apply this only to lexicon words with low frequency.
			See Michael Collins' NLP Coursera notes, chap.2
			:rtype: str
		"""
		if len(word) > 1 and word[0].isupper() and word[1] == '.':
			return '!capPeriod!'  # Person's name initial i.e. "M."

		if word.isupper():
			return '!allCaps!'

		if word.istitle():
			return '!initCap!'  # TODO: should distinguish between words at beginning of sentence?

		if word.islower():
			return '!lowercase!'

		return "!other"  # weird punctuation etc


class Ratnaparkhi96Features(SequenceFeaturesTemplate):
	""" Features idenetical to Ratnaparkhi's 1996 paper with a few more suffixes
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
			if not '' in args and not None in args:
				features.append(' '.join((name,) + tuple(args)))

		def add_suffixes(feature, word, n):
			if not Context.is_pseudo(word):  # i.e. it's a pseudo word/class
				for s in cls.get_suffixes(word, n):
					add(feature, s, tag)

		features = []  # Should be a Set but that's too slow :(
		words, tags = context
		tag = tags[i]  # tag we're predicting
		n_suffixes = 4

		add('i word', words[i], tag)
		add('i-1 word', words[i-1], tag)
		add('i-2 word', words[i-2], tag)
		add('i+1 word', words[i+1], tag)
		add('i+2 word', words[i+2], tag)

		# Bigram's
		add('i-2 word,i-1 word', words[i-2], words[i-1], tag)
		add('i-1 word,i word', words[i-1], words[i], tag)
		add('i word,i+1 word', words[i], words[i+1], tag)
		add('i+1 word,i+2 word', words[i+1], words[i+2], tag)

		# Current Tag
		add('i tag ', tag)

		# Bigram tags (skip-grams?)
		add('i-1 tag, i tag', tags[i-1], tag)
		add('i-2 tag, i tag', tags[i-2], tag)
		add('i+1 tag, i tag', tags[i+1], tag)
		add('i+2 tag, i tag', tags[i+2], tag)

		# Tri-gram tags
		add('i-2 tag,i-1 tag,i tag', tags[i-2], tags[i-1], tag)
		add('i-1 tag,i tag,i+1 tag', tags[i-1], tag, tags[i+1])
		add('i tag,i+1 tag,i+2 tag', tag, tags[i+1], tags[i+2])

		add_suffixes('i word suffix', words[i], n_suffixes)

		return features


class Context(object):
	"""
	The main idea of this class is to reduce the number of times feature_templates.get() and zip(*sequence) are called.
	"""
	# Symbol added to the beginning and end of the sequence so we don't fall of the edges """
	BEGIN_SYMBOL = '!START!'
	END_SYMBOL = '!END!'

	def __init__(self, sequence, feature_templates=Ratnaparkhi96Features):
		"""
		:param sequence: List of tuples of the form (word, label) or a tuple of 2 lists (words, labels)
		:param feature_templates: A SequenceFeaturesTemplate object
		:return:
		"""
		extra = feature_templates.LOOK_BEHIND
		BEGIN = list(reversed([Context.BEGIN_SYMBOL * i for i in range(1, extra+1)]))
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

	@classmethod
	def is_pseudo(cls, thing):
		""" Is it a START/END tag or a Normalised class
		:param thing: String to check
		:return: Boolean
		"""
		return True if thing[-1] != '!' and thing[0] != '!' else False

class ForwardBackward(object):
	""" The forward-backward algorithm can be used to find the most likely state for any point in time. It cannot,
		however, be used to find the most likely sequence of states (see Viterbi algorithm).
	"""

	def __init__(self, model):
		self.model = model  # needs to implement interface: potential(state_from, state_to, context_index)

	def p(self, state_from, state_to, seq, i):
		context  = Context((seq, [''] * len(seq)))
		return self.model.potential(state_to, state_from, context, i)

	def forward_backward(self, unlabeled_sequence, states):
		""" Run the forwards-backwards algorithm
		:param context: Context object
		:param states: list of all possible states/tags
		:return:
		"""
		# TODO We can do better initial states here but need start tokens from Context object
		start = {}
		for st in states:
			tags = ['']*len(unlabeled_sequence)
			tags[0] = st
			context = Context((unlabeled_sequence, tags))
			start[st] = self.model.probabilities(0, context)[st]
		end = {}
		for st in states:
			m = len(unlabeled_sequence)
			tags = [''] * m
			tags[m-1] = st
			context = Context((unlabeled_sequence, tags))
			end[st] = self.model.probabilities(m-1, context)[st]

		return self.fwd_bkw(context, states, start, end)

	def fwd_bkw(self, context, states, start, end):
		""" Iterative version of the Forward-Backwards algorithm
			Taken from Wikipedia
		:param context: observation sequence (sentence), this is a Context object.
		:param states: states
		:param start: start probability
		:param end: end state
		:return: forward and backward probabilities + posteriors
		"""
		seq = context.words
		m = len(seq)

		# Forward part of the algorithm
		# the probability of ending up in any particular state given the first k observations in the sequence

		fwd = []
		f_prev = {}
		for i, _ in enumerate(seq):
			f_curr = {}
			for st in states:
				if i == 0:
					# base case for the forward part
					f_curr = start
					break
				else:
					f_curr[st] = sum(f_prev[k] * self.p(k, st, seq, i) for k in states)
			f_curr = normalize(f_curr)
			# iterate (instead of recurs)
			fwd.append(f_curr)
			f_prev = f_curr
		print pandas.DataFrame(fwd).to_string()
		#z = sum([fwd[m-1][k] for k in states])  # normalizer for the posteriors

		# Backward part of the algorithm
		# the probability of observing the remaining observations given any starting point k

		bkw = []
		b_prev = {}
		# for i, x_i_plus in enumerate(reversed(seq[1:] + [None])):
		for i in range(m-1, 0, -1):
			b_curr = {}
			for st in states:
				if i == m-1:
					# base case for backward part
					b_curr = end
					break
				else:
					b_curr[st] = sum([self.p(st, l, seq, i) * b_prev[l] for l in states])
			b_curr = normalize(b_curr)
			# iterate
			bkw.insert(0, b_curr)
			b_prev = b_curr

		print pandas.DataFrame(bkw).to_string()
		#p_bkw = sum(bkw[0][l] for l in states) # Transition to first state?

		# merging the two parts
		posterior = []
		for i in range(m):
			posterior.append({st: fwd[i][st] * bkw[i][st] for st in states})
		posterior = normalize(posterior)
		print pandas.DataFrame(posterior).to_string()
		# assert z == p_bkw
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
			for s in all_states:
				context = Context(list(izip_longest(seq, path[s], fillvalue='')))
				# We only consider labels we have seen for this word (see: get_labels(word))
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
