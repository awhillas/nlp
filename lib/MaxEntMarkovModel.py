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
import pandas
from os import path
import cPickle as pickle

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

		return [word[-i:] for i in range(1, n+1)]


class WordNormaliser(object):
	"""
	Handles normalisation of words. Used to close vocabulary.
	"""

	def __init__(self, data):
		self._data = data
		self._index = 0
		pass

	# iterator behavior

	def __iter__(self):
		return self

	def next(self):
		if self._index >= len(self._data):
			raise StopIteration
		words, labels = map(list, zip(*self._data[self._index]))
		self._index += 1
		return zip(self.sentence(words), labels)

	def __next__(self):
		return next()

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

class MaxEntMarkovModel(SequenceModel):
	""" Maximum Entropy Markov Model
		What makes it Markov is that the previous two labels are part of the
		features which are passed on via th context.
	"""

	def __init__(self, feature_templates, word_normaliser):
		"""
		:param feature_templates: Instance of SequenceFeaturesTemplate
		:param word_normaliser: Instance of WordNormaliser
		:param regularization_parameter: Parameter used to tune the regularization amount.
		:return:
		"""
		super(MaxEntMarkovModel, self).__init__()
		self.feature_templates = feature_templates
		self.normaliser = word_normaliser
		# self.learnt_features = SortedDict()  # all features broken down into counts for each label
		self.learnt_features = SortedDict()  # full features including labels
		self.weights = SortedDict()  # lambdas aka feature weights aka model parameters
		self.total = 0  # Total words seen in training corpus
		self.tag_count = {}  # Keep a count of each tag
		self.word_tag_count = {}  # Keep track of word -> tag -> count
		self.tagdict = {}  # To be used for fast lookup of unambigous words

	@classmethod
	def save_file(cls, save_dir=None, filename_prefix = ''):
		return path.join(save_dir, "MaxEntMarkovModel" + filename_prefix + ".pickle")

	def save(self, save_dir=None, filename_prefix = ''):
		if save_dir is None:
			save_dir = path.join(path.dirname(__file__))
		print "Saving MaxEntMarkovModel:", self.save_file(save_dir, filename_prefix)
		with open(self.save_file(save_dir, filename_prefix), 'wb') as f:
			pickle.dump(self.__dict__, f, -1)
			print "Model saved!"

	def load(self, save_dir=None, filename_prefix = ''):
		if save_dir is None:
			save_dir = path.join(path.dirname(__file__))
		file_name = self.save_file(save_dir, filename_prefix)
		if path.exists(file_name):
			print "Loading MaxEntMarkovModel:", self.save_file(save_dir, filename_prefix)
			with open(file_name) as f:
				self.__dict__.update(pickle.load(f))
			print "Model loaded!"
			return True
		else:
			raise Exception("MaxEntMarkovModel not loaded! File does not exist? '%s'" % file_name)

	def get_labels(self, word):
		""" Use tag dict. of all seen words. """
		if word in self.word_tag_count:
			return self.word_tag_count[word].keys()
		else:
			return self.tag_count.keys()

	def guess_tag(self, word):
		""" Use the more selective tagdict of seen tags. """
		guess = self.tagdict.get(word)
		return [guess] if guess else self.tag_count.keys()

	def guess_tags(self, sentence):
		tags = []
		for word in sentence:
			guess = self.tagdict.get(word)
			tags.append(guess) if guess else tags.append('')
		return tags

	def get_classes(self):
		return self.tag_count.keys()

	def train(self, data, regularization=0.33, maxiter=1, optimise=True):
		"""
		:param data: List of sentences, Sentences are lists if (word, label) sequences.
		:return: true on success, false otherwise
		"""
		for sentence in self.normaliser.all(data):
			context = Context(sentence, self.feature_templates)
			for i, word in enumerate(context.words):
				label = context.labels[i]
				self.add_tag(word, label)
				for f in context.get_features(i, label):
					self.learnt_features.setdefault(f, 0)
					self.learnt_features[f] += 1
					self.weights.setdefault(f, 1.0)  # initial default to 1.0
		self._make_tagdict()
		if optimise:
			return self._learn_parameters(data, regularization, maxiter)
		else:
			return False

	@classmethod
	def _merge_weight_values(cls, features, values):
		return SortedDict(izip(features, values))

	def _learn_parameters(self, data, regularization, maxiter):
		""" Learn the parameter vector (weights) for the model
		:param data: List of sentences, Sentences are lists if (word, label) sequences.
		:return: True on success, False otherwise
		"""
		print "Estimating parameters"

		normaised_data = self.normaliser.all(data)

		def objective(x):  # Objective function
			v = self._merge_weight_values(self.weights.iterkeys(), x)
			log_p = 0.0
			for seq in normaised_data:
				context = Context(seq, self.feature_templates)
				for i, (_, label) in enumerate(seq):
					probabilities = self.probabilities(i, context, v)
					if probabilities[label] > 0.0:  # Getting zero some times...?
						log_p += log(probabilities[label])

			# Regularization
			regulatiser = sum([param * param for param in v.itervalues()]) * (regularization / 2)

 			print "{:>13.2f} - {:>13.2f} = {:>+13.2f} (max:{:>+7.2f}, min:{:>+7.2f})".format(log_p, regulatiser, log_p - regulatiser, max(x), min(x))
			return log_p - regulatiser

		def inverse_gradient(x):
			""" Inverse (coz we want the max. not min.) of the Gradient of the objective. """
			v = self._merge_weight_values(self.weights.iterkeys(), x)
			dV = SortedDict.fromkeys(self.weights.iterkeys(), 0.0)  # gradient vector output

			# Expected/predicted feature counts

			for n, seq in enumerate(normaised_data):
				context = Context(seq, self.feature_templates)  # build all features for all positions in seq.
				for i, _ in enumerate(seq):
					probabilities = self.probabilities(i, context, v)  # get prob. for all labels at position i
					for label in self.tag_count.iterkeys():
						for feature in context.get_features(i, label):  # merge features with label at position i
							if feature in v:
								dV[feature] += probabilities[label]

			# Actual feature counts + regularize
			# Assumption: that the param. training is over the same data set as the feature extraction
			for f, count in self.learnt_features.iteritems():
				dV[f] -= count
				dV[f] += v[f] * regularization

			return array(dV.values())

		# Maximise, actually.
		params = self.weights.values()
		bnds = [(-20, 20)] * len(params)  # upper and lower for each var max python can exp(*) without overflow
		if len(params) > 0:
			result = minimize(fun=lambda x: -objective(x), jac=lambda x: inverse_gradient(x), x0=params,\
							  method='L-BFGS-B', options={'maxiter': maxiter}, bounds=bnds)
		else:
			print "No parameters to optimise!?"
			return False

		self.weights = SortedDict(izip(self.learnt_features.iterkeys(), result.x.tolist()))
		if not result.success:
			print result.message
		return True

	def frequency_tag(self, unlabeled_sequence):
		""" Label with the highest frequency word label (base line)
		:param unlabeled_sequence: List of sentences, which are lists of words.
		:return: List if labeled sentences: lists of (word) tuple pairs.
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
		sentence = self.normaliser.sentence(seq)
		all_tags = self.tag_count.keys()
		# Try and guess the tags for all the words first to save time and increase the features.
		tags = self.guess_tags(sentence)
		context = Context((sentence, tags), self.feature_templates)
		if tags[0]:
			start_p = dict.fromkeys(all_tags, 0.01/(len(all_tags)-1))
			start_p[tags[0]] = 0.99
		else:
			start_p = dict([(t, self.potential(t, None, context, 0)) for t in all_tags])
		prob, tags = Viterbi.viterbi(sentence, all_tags, start_p, self)
		return tags

	def tag(self, sentence):
		return self.label(sentence)

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
		if i > 0:
			tags[i-1] = prev_label
		# Need to recreate a new Context with the new tags so the features get regenerated.
		return self.probabilities(i, Context((context.words, tags), self.feature_templates), self.weights)[label]

	def potential_backward(self, prev_label, label, context, i):
		""" Same as potential() but we're going in the other direction for the backwards part of the forward-backwards
			algo.
		"""
		tags = context.labels
		tags[i] = label
		if i > 0:
			tags[i+1] = prev_label
		# Need to recreate a new Context with the new tags so the features get regenerated.
		return self.probabilities(i, Context((context.words, tags), self.feature_templates), self.weights)[label]

	def tag_probability_distributions(self, context):
		fb = ForwardBackward(self)
		return fb.forward_backward(context, self.tag_count.keys())

	def multi_tag(self, unlabeled_sequence, ambiguity = 0.0):
		tags = []
		normalised_sentence = self.normaliser.sentence(unlabeled_sequence)
		context = Context((normalised_sentence, self.guess_tags(normalised_sentence)))
		distros = self.tag_probability_distributions(context)  # tag probability distributions for each word
		tags = self.threshold(distros, ambiguity) if ambiguity > 0.0 else distros
		show(tags, unlabeled_sequence)
		return tags

	@classmethod
	def threshold(cls, tag_probability_distributions, ambiguity):
		""" Threshold based on a given ambiguity level which is a percentage of the highest probability in the distribution.
		:param tag_probability_distribution: list of dicts which have tags as keys and probabilities as values
		:param ambiguity: float between 0 and 1
		:return: {tag: prob} dict
		"""
		tags = []
		for i, tag_probs in enumerate(tag_probability_distributions):
			c_max = max(tag_probs.iterkeys(), key=(lambda key: tag_probs[key]))
			top_tags = dict( (c,p) for c, p in tag_probs.iteritems() if c == c_max or p > (ambiguity * tag_probs[c_max]) )
			tags.append(top_tags)
		return tags

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
			v = self.weights

		class_probabilities = dict()
		for label in self.tag_count.iterkeys():
			class_probabilities.setdefault(label, 1.0)  # coz exp(0) = 1
			for feature in context.get_features(i, label):
				if feature in v:
					class_probabilities[label] *= exp(v[feature])  # exp(a+b) = exp(a) * exp(b)
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

	def _make_tagdict(self):
		"""Make a tag dictionary for single-tag words."""
		freq_thresh = 10  # doesn't scale to different data set sizes :(
		ambiguity_thresh = 0.97
		for word, tag_freqs in self.word_tag_count.items():
			tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
			n = sum(tag_freqs.values())
			# Don't add rare words to the tag dictionary
			# Only add quite unambiguous words
			if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
				self.tagdict[word] = tag


class CollinsNormalisation(WordNormaliser):
	""" Normalisations taken from Michel Collins notes
		'Chapter 2: Tagging problems and Hidden Markov models'
		TODO: make normaiser iterable to save memoery.
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
		out = cls.web_entites(word)
		if out == word:
			out = cls.pseudo_map_digits(out)
		if out == word:
			out = cls.junk(word)
		if out == word:
			return out.lower()  # is this a good idea?
		else:
			return out

	@classmethod
	def web_entites(cls, word):
		if word in cls.EMOS:
			# print '!emoticon!', word
			return '!emoticon!'
		if bool(cls.EMAIL_REGEX.search(word.lower())) \
				or word.count('@') == 1:
			# print '!email!', word
			return '!email!'
		if bool(cls.URL_REGEX.search(word)):
			#print '!url!', word
			return '!url!'
		return word

	@classmethod
	def junk(cls, word):
		if len(word) > 1 \
				and "'" not in word \
				and ('-' not in word or word.count('-') > 2):
			if all(i in string.punctuation for i in word):
				return '!allPunctuation!'
			if not all(i in string.letters for i in word):
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
					# print '!mixedUp!', word
					return '!mixedUp!'
		return word

	@classmethod
	def pseudo_map_digits(cls, word):
		""" Map a word to a pseudo word with digits.
			Want to apply this to all input words.
			See Michael Collins' NLP Coursera notes, chap.2
			:rtype: str
		"""
		if bool(cls.RE_DIGITS.search(word)):  # contains digits
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
			Want to apply this only to lexicon words with low frequency or unseen words.
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
		:return: list of features WITHOUT the target tag. Need to add that to these before use.
		"""

		def add(name, *args):
			if not '' in args and not None in args:
				features.append(' '.join((name,) + tuple(args)))

		def add_suffixes(name, word, n):
			if word is not None and  not Context.is_pseudo(word):  # i.e. it's a pseudo word/class
				for s in cls.get_suffixes(word, n):
					add(name, s)

		features = []  # Should be a Set but that's too slow :(
		words, tags = context
		tag = tags[i]  # tag we're predicting
		n_suffixes = 4

		#add('bias')  # Acts like a prior

		add('i word', words[i])
		add('i-1 word', words[i-1])
		add('i-2 word', words[i-2])
		add('i+1 word', words[i+1])
		add('i+2 word', words[i+2])

		# Bigram's
		add('i-2 word, i-1 word', words[i-2], words[i-1])
		add('i-1 word, i word', words[i-1], words[i])
		add('i word, i+1 word', words[i], words[i+1])
		add('i+1 word, i+2 word', words[i+1], words[i+2])

		# Current Tag
		#add('i tag ')  # does this add too much weight to more frequent tags?

		# Bigram tags (skip-grams?)
		add('i-1 tag', tags[i-1])
		add('i-2 tag', tags[i-2])
		add('i+1 tag', tags[i+1])
		add('i+2 tag', tags[i+2])

		# Tri-gram tags
		add('i-2 tag, i-1 tag', tags[i-2], tags[i-1])
		add('i-1 tag, i+1 tag', tags[i-1], tags[i+1])
		add('i+1 tag, i+2 tag', tags[i+1], tags[i+2])

		add_suffixes('i word suffix', words[i], n_suffixes)

		return features


class OrthographicFeatures(SequenceFeaturesTemplate):
	# TODO: add word class's from:
	# 	Originally: Michael Collins (2002) Ranking algorithms for named-entity extraction: Boosting and the voted perceptron
	#	Explained in: Burr Settles (2004) Biomedical Named Entity Recognition Using Conditional Random Fields and Rich Feature Sets

	# "Words are also assigned a generalized “word class” similar to Collins (2002), which replaces
	# capital letters with ‘A’, lowercase letters with ‘a’, digits with ‘0’, and all other characters
	# with ‘_’. There is a similar “brief word class” feature which collapses consecutive identical
	# characters into one. Thus the words “IL5” and “SH3” would both be given the features
	# WC=AA0 and BWC=A0, while “F-actin” and “T-cells” would both be assigned WC=A_aaaaa and BWC=A_a."

	@classmethod
	def word_class(self, word):
		wc = ''
		for l in word:
			if l in string.lowercase:
				wc += 'a'
			elif l in string.uppercase:
				wc += 'A'
			elif l in string.digits:
				wc += '0'
			else:
				wc += '_'
		return wc

	@classmethod
	def brief_word_class(self, word_class):
		return re.sub(r'(.)\1+', r'\1', word_class)


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
		self.features = []  # feature templates for each word position in the sentence
		for i, _ in enumerate(self.sequence):
			self.features.append([f for f in feature_templates.get(i+extra, (BEGIN+list(self.words)+END, BEGIN+list(self.labels)+END))])

	def __len__(self):
		return len(self.words)

	def get_features(self, i, label):
		return [intern(str(f+" &"+label)) for f in self.features[i]]  # merge the feature set with the label

	@classmethod
	def is_pseudo(cls, thing):
		""" Is it a START/END tag or a Normalised class
		:param thing: String to check
		:return: Boolean
		"""
		return True if thing[-1] == '!' and thing[0] == '!' else False


class ForwardBackward(object):
	""" The forward-backward algorithm can be used to find the most likely state for any point in time. It cannot,
		however, be used to find the most likely sequence of states (see Viterbi algorithm).
	"""

	def __init__(self, model):
		self.model = model  # needs to implement interface: potential(state_from, state_to, context_index)

	def p(self, state_from, state_to, seq, i, forwards=True):
		context  = Context((seq, [''] * len(seq)))  # TODO: this works in forwards but not backwards + should guess_tags()\
		if forwards:
			return self.model.potential(state_to, state_from, context, i)
		else:
			return self.model.potential_backward(state_to, state_from, context, i)

	def forward_backward(self, context, states):
		""" Run the forwards-backwards algorithm
		:param context: Context object
		:param states: list of all possible states/tags
		:return:
		"""
		def get_edge_prob(i):
			edge = {}
			if context.labels[i]:
				edge = dict.fromkeys(states, 0.0)  # dangerous to have zeros in the pipe
				edge[context.labels[i]] = 1.0
			else:
				for st in states:
					edge[st] = self.model.probabilities(i, context)[st]
			return edge

		m = len(context.words) - 1
		start = get_edge_prob(0)  # Calculate start states
		end = get_edge_prob(m)  # Calculate end states

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
		def force(tag):
			dist = dict.fromkeys(states, 0.01/(len(states)-1))
			dist[tag] = 0.99
			return dist

		seq = context.words
		tags = context.labels
		m = len(seq)-1

		# Forward part of the algorithm
		# the probability of ending up in any particular state given the _first_ k observations in the sequence

		fwd = []
		f_prev = {}
		for i, _ in enumerate(seq):
			if not tags[i]:
				f_curr = {}
				for st in states:
					if i == 0:
						# base case for the forward part
						f_curr = start
						break
					else:
						# f_curr[st] = sum(f_prev[k] * self.p(k, st, seq, i) for k in states)
						# f_curr[st] = sum(f_prev[k] * self.p(k, st, seq, i) for k in self.model.get_labels(seq[i-1]))  # speed up
						f_curr[st] = sum(f_prev[k] * self.p(k, st, seq, i) for k in self.model.guess_tag(seq[i-1]))  # speed up
				f_curr = normalize(f_curr)
			else:
				f_curr = force(tags[i])  # if we are certain about a particular tag then force it  skip the DP monkey work.
			# iterate (instead of recurs)
			fwd.append(f_curr)
			f_prev = f_curr

		# show(fwd, seq, "Forward:")

		# Backward part of the algorithm
		# the probability of observing the _remaining_ observations given any starting point k
		# TODO Backwards pass seems to produce a flat prob. distribution :(

		bkw = []
		b_prev = {}
		# for i, x_i_plus in enumerate(reversed(seq[1:] + [None])):
		for i in range(m, -1, -1):
			if not tags[i]:
				b_curr = {}
				for st in states:
					if i == m:
						# base case for backward part
						b_curr = end
						break
					else:
						# b_curr[st] = sum([self.p(st, l, seq, i) * b_prev[l] for l in states])
						# b_curr[st] = sum([self.p(st, l, seq, i, False) * b_prev[l] for l in self.model.get_labels(seq[i+1])])  # is seq[i] right..?
						b_curr[st] = sum([self.p(st, l, seq, i, False) * b_prev[l] for l in self.model.guess_tag(seq[i+1])])  # is seq[i] right..?
				b_curr = normalize(b_curr)
			else:
				b_curr = force(tags[i])  # again, skip the monkey work if we already know the tag.
			bkw.insert(0, b_curr)
			b_prev = b_curr  # iterate

		# show(bkw, seq, "Backwards:")

		# merging the two parts
		posterior = []
		for i in range(m+1):
			posterior.append(normalize({st: fwd[i][st] * bkw[i][st] for st in states}))

		# show(posterior, seq, "Posteriors")
		# return fwd, bkw, posterior
		return posterior

def show(matrix, seq, name=''):
	pd = pandas.DataFrame(matrix).transpose()
	pd.columns = seq
	print name, "\n", pd.to_string()


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
		def force(tag):
			dist = dict.fromkeys(states, 0.01/(len(states)-1))
			dist[tag] = 0.99
			return dist

		V = [{}]  # len(seq) x len(states) dynamic programing table
		path = {}  # back pointers

		# Initialize base cases (t == 0)

		for s in all_states:
			V[0][s] = start_p[s]
			path[s] = [s]

		guesses = model.guess_tags(seq)

		# Run Viterbi for j > 0

		for j in range(1, len(seq)):
			V.append(dict.fromkeys(all_states, 0))
			new_path = {}
			for s in all_states:
				context = Context(list(izip_longest(seq, path[s], fillvalue='')))
				# We only consider labels we have seen for this word (see: guess_tag(word)) or all if unseen.
				(prob, prev_state) = max( (V[j - 1][s0] * model.potential(s, s0, context, j), s0) for s0 in model.guess_tag(seq[j-1]) )
				V[j][s] = prob
				new_path[s] = path[prev_state] + [s]
			# Don't need to remember the old paths
			path = new_path

		# Find the max sequence

		n = 0  # if only one element is observed max is sought in the initialization values
		if len(seq) != 1:
			n = j
		#print_dptable(V, seq)
		(prob, prev_state) = max((V[n][y], y) for y in all_states)
		return prob, path[prev_state]
