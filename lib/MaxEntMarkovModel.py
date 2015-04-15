__author__ = 'alex'

import re
import string
from math import exp, log, fsum
from collections import defaultdict
from sortedcontainers import SortedDict  # see http://www.grantjenks.com/docs/sortedcontainers/sorteddict.html
from scipy.optimize import minimize
from itertools import izip
from scipy import array

# Interfaces
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SequenceModel():
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


class SequenceFeaturesTemplate():
	""" Interface to features used in sequencing models like MaxEnt or Log Linear etc
	"""

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


class WordNormaliser():
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
		self.feature_templates = feature_templates
		self.normaliser = word_normaliser
		self.normalised_data = self.normaliser.all(data)
		self.regularization_parameter = regularization_parameter
		self.learnt_features = SortedDict()  # all features broken down into counts for each label
		self.learnt_features_full = SortedDict()  # full features including labels
		self.parameters = SortedDict()  # lambdas aka weights aka model parameters
		self.tag_count = {}  # Keep a count of each tag
		self.word_tag_count = {}  # Keep track of word -> tag -> count

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
					self.learnt_features[f][
						label] += 1  # Keep counts of features by tag for gradient. See: learn_parameters()
				for f in context.get_features(i, label):
					self.learnt_features_full.setdefault(f, 0)
					self.learnt_features_full[f] += 1
					self.parameters.setdefault(f, 1.0)  # initial default to 1.0

	def learn_parameters(self, data=None):
		""" Learn the parameter vector (weights) for the model
		:param data: List of sentences, Sentences are lists if (word, label) sequences.
		:return: True on success, False otherwise
		"""
		print "Estimating parameters"

		if data is None:
			data = self.normalised_data

		def objective(x):  # Objective function
			v = dict(izip(self.parameters.iterkeys(), x))
			log_p = 0.0
			for seq in data:
				for i, (word, label) in enumerate(seq):
					probabilities = self.probabilities(i, Context(seq, self.feature_templates), v)
					log_p += log(probabilities[label])

			# Regularization
			regulatiser = sum([param * param for param in v.itervalues()]) * self.regularization_parameter / 2

			print log_p, regulatiser, log_p - regulatiser
			return log_p - regulatiser

		def inverse_gradient(x):
			""" Inverse (coz we want the max. not min.) of the Gradient of the objective
			"""
			v = dict(izip(self.parameters.iterkeys(), x))  # current param. vector
			dV = dict.fromkeys(self.parameters.iterkeys(), 0.0)  # gradient vector output

			# Predicted counts

			for n, seq in enumerate(data):
				print "#", n, (float(n) + 1) / len(data) * 100, "%"
				context = Context(seq, self.feature_templates)

				for i, _ in enumerate(seq):
					probabilities = self.probabilities(i, context, v)

					# Expected/predicted feature counts

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

		result = minimize(fun=lambda x: -objective(x), jac=lambda x: inverse_gradient(x), x0=self.parameters.values(), method='L-BFGS-B', options={'maxiter': 20})  # Maximise, actually

		if result.success:
			# return result.x
			self.parameters = result.x
			return True
		else:
			raise RuntimeError('Error learning parameters: ' + result.message)

	def label(self, sequences):
		""" Prediction: Calls the Viterbi algorithm to label each sentence in the input sequence
		:param sequences: List of sentences, which are lists of words.
		:return: List if labeled sentences: lists of (word, tag) tuple pairs.
		"""
		for sentence in input:
			words, tags = zip(*sentence)
			for i, word in enumerate(words):
				# TODO: replace this with the Viterbi or Froward-Backward algorthm
				probabiities = dict([(label, self.p(label, (words, i), self.parameters)) for label in self.tag_count.keys()])

	def probabilities(self, i, context, v):
		""" Gets the probability distribution for all the tags/classes/labels for the current word/item in the
			sentence/sequence with a given feature.
			aka The Soft Max function
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
		return self.normalize(class_probabilities)

	@classmethod
	def normalize(cls, d, target=1.0):
		""" Make all the values add to target
		:param d: dict
		:param target: target sum, defaults to 1.0
		:return: dict
		"""
		raw = fsum(d.itervalues())
		factor = target / raw
		return {key: value * factor for key, value in d.iteritems()}

	def add_tag(self, w, t):
		""" Keep track of words and their tags
		:param w: word
		:param t: tag/label
		"""
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


class HonibbalsFeats(SequenceFeaturesTemplate):
	""" Features nicked from Mathew Honnibal's PerceptronTagger for TextBlob
	"""

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

		features = []  # Should be a set but that's too slow :(
		words, tags = context
		# print i, words, tags

		add('bias')  # It's useful to have a constant feature, which acts sort of like a prior
		add('i word', words[i])
		add('i suffix', words[i][-3:])
		add('i pref1', words[i][0])

		if i >= 1:
			add('i-1 word', words[i - 1])
			add('i-1 suffix', words[i - 1][-3:])  # TODO: better suffixes
			add('i-1 tag', tags[-1])
			add('i-1 tag+i word', tags[i - 1], words[i])
		else:
			add('i-1 tag', '*')
			add('i-1 tag+i word', '*', words[i])

		if i >= 2:
			add('i-2 word', words[i - 2])
			add('i-2 tag', tags[i - 2])
			add('i tag+i-2 tag', tags[i - 1], tags[i - 2])
		else:
			add('i-2 tag', '**')
			add('i tag+i-2 tag', '*', '**')

		if i + 1 < len(words):
			add('i+1 word', words[i + 1])
			add('i+1 suffix', words[i + 1][-3:])

		if i + 2 < len(words):
			add('i+2 word', words[i + 2])

		return features


class Context():
	"""
	The main idea of this class is to reduce the number of times feature_templates.get() and zip(*sequence) are called.
	"""

	def __init__(self, sequence, feature_templates):
		self.sequence = sequence
		self.words, self.labels = zip(*sequence)
		self.features = [[f for f in feature_templates.get(i, (self.words, self.labels))] for i, _ in enumerate(sequence)]

	def get_features(self, i, label):
		return [f + " " + label for f in self.features[i]]  # merge the feature set with the label


class Viterbi():
	@classmethod
	def viterbi(cls, obs, states, start_p, trans_p, emit_p):
		""" The Viterbi algorithm.
			assume that the observation sequence obs is non-empty and that
			trans_p[i][j] and emit_p[i][j] is defined for all states i,j.
			Taken from https://en.wikipedia.org/wiki/Viterbi_algorithm
		:param obs: the sequence of observations
		:param states: the set of hidden states
		:param start_p: start probability
		:param trans_p: the transition probabilities
		:param emit_p: the emission probabilities
		:return:
		"""
		V = [{}]
		path = {}

		# Initialize base cases (t == 0)
		for y in states:
			V[0][y] = start_p[y] * emit_p[y][obs[0]]
			path[y] = [y]

		# Run Viterbi for t > 0
		for t in range(1, len(obs)):
			V.append({})
			newpath = {}

			for y in states:
				(prob, state) = max((V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
				V[t][y] = prob
				newpath[y] = path[state] + [y]

			# Don't need to remember the old paths
			path = newpath
		n = 0  # if only one element is observed max is sought in the initialization values
		if len(obs) != 1:
			n = t
		cls.print_dptable(V)
		(prob, state) = max((V[n][y], y) for y in states)
		return (prob, path[state])

	@classmethod
	def print_dptable(V):
		s = "	 " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
		for y in V[0]:
			s += "%.5s: " % y
			s += " ".join("%.7s" % ("%f" % v[y]) for v in V)
			s += "\n"
		print(s)
