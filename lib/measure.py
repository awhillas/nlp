__author__ = "Alexander Whillas <whillas@gmail.com>"

from sortedcontainers import SortedDict
import re
import csv
import StringIO
from ascii_graph import Pyasciigraph

class Measure(object):
	""" Class for scoring binary classification models
	"""
	def __init__(self):
		self.true_positive = 0
		self.true_negative = 0
		self.false_positive = 0
		self.false_negative = 0

	def total(self):
		out = self.true_positive + self.true_negative + self.false_positive + self.false_negative
		assert out >= 0 and isinstance(out, int), "Can only add positive integers. Total was %d" % out
		return out

	def tp(self, add=1):
		""" True Positive
		:param add: How many to add to the tally
		"""
		assert add >= 0 and isinstance(add, int), "Can only add positive integers. Received %d" % add
		self.true_positive += add

	def tn(self, add=1):
		""" True Negative
		:param add: How many to add to the tally
		"""
		assert add >= 0 and isinstance(add, int), "Can only add positive integers. Received %d" % add
		self.true_negative += add

	def fp(self, add=1):
		""" False Positive
		:param add: How many to add to the tally
		"""
		assert add >= 0 and isinstance(add, int), "Can only add positive integers. Received %d" % add
		self.false_positive += add

	def fn(self, add=1):
		""" False Negative
		:param add: How many to add to the tally
		"""
		assert add >= 0 and isinstance(add, int), "Can only add positive integers. Received %d" % add
		self.false_negative += add

	def precision(self):
		return float(self.true_positive) / (self.true_positive + self.false_positive)

	def recall(self):
		return float(self.true_positive) / (self.true_positive + self.false_negative)

	def accuracy(self):
		return float(self.true_positive + self.true_negative) / self.total()

	def f1score(self):
		""" Harmonic mean """
		return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

	def __unicode__(self):
		return "Accuracy: %.2f; F1: %.2f; Recall: %.2f; Precision: %.2f" \
		       % (self.accuracy(), self.f1score(), self.recall(), self.precision())

	def __str__(self):
		return unicode(self).encode('utf-8')

class ConfusionMatrix(object):
	"""
	A specific table layout that allows visualization of the performance of an algorithm.
	Each column of the matrix represents the instances in a predicted class, while each row represents the instances in
	an actual class.
	"""
	def __init__(self, classes):
		self.table = SortedDict.fromkeys(classes, SortedDict)
		for key, _ in self.table.iteritems():
			self.table[key] = SortedDict.fromkeys(classes, 0)

	def add(self, predicted, actual):
		self.table.setdefault(actual, SortedDict())
		self.table[actual].setdefault(predicted, 0)
		self.table[actual][predicted] += 1

	def show(self):
		row_format = "{:>5} " * (len(self.table) + 1)
		return self.get_output(row_format)

	def csv(self):
		output = StringIO.StringIO()
		wr = csv.writer(output, quoting=csv.QUOTE_ALL)
		# Header
		wr.writerow([""] + list(self.table.keys()))
		# Rows
		for tag, values in self.table.iteritems():
			data = [str(v) if v > 0 else '' for v in values.values()] + [sum(values.values()) - values[tag]]
			wr.writerow([tag] + data)
		return output.getvalue()

	def get_output(self, row_format):
		out = ''
		# Header
		out += row_format.format("", *[re.escape(s) for s in self.table.keys()])
		# Rows
		for tag, values in self.table.iteritems():
			data = [str(v) if v > 0 else '' for v in values.values() ] + [sum(values.values())]
			out += row_format.format( tag, *data )
		return out

	def precision(self):
		correct = 0
		total = 0
		for actual, row in self.table.iteritems():
			for predicted, count in row.iteritems():
				if actual == predicted:
					correct += count
				total += count
		return float(correct) / total

	def compare(self, predicted_sequences, expected_sequences):

		sentence_errors = word_error = word_count = 0

		for i, gold_seq in enumerate(expected_sequences):
			words, gold_labels = zip(*gold_seq)
			words2, predicted_labels = zip(*predicted_sequences[i])
			error = False
			for j, word in enumerate(words2):
				word_count += 1
				if word == words[j]:
					self.add(gold_labels[j], predicted_labels[j])
					if gold_labels[j] != predicted_labels[j]:
						error = True
						word_error += 1
				else:
					print "Sequences out of sync", words, words2
					raise
			if not error:
				sentence_errors += 1

		return word_count, word_error, sentence_errors

class POSTaggerMeasure(object):
	""" Class to measure the performance of a POS Tagger. """
	def __init__(self, classes):
		self.matrix = ConfusionMatrix(classes)
		self.sent_length_totals = {}
		self.sent_correct = {}

	def test(self, words, predicted, gold, verbose = False):
		error_count = 0
		sentence_error = False
		length = len(words)
		self.sent_length_totals.setdefault(length, 0)
		self.sent_length_totals[length] += 1

		for j, w in enumerate(words):
			self.matrix.add(gold[j], predicted[j])
			if predicted[j] != gold[j]:
				error_count += 1
				sentence_error = True

		if not sentence_error:
			self.sent_correct.setdefault(length, 0)
			self.sent_correct[length] += 1

		if verbose:
			POSTaggerMeasure.print_solution(words, predicted, gold)
			correct = len(words) - error_count
			print "Correct:", correct, "/", len(words), ", rate:", "%.1f" % (float(correct) / len(words) * 100), "%"

	def print_tally(self):
		print "Tag:", "{:>4.2f}".format(self.matrix.precision() * 100), "%"
		print "Sentence: ", "{:>4.2f}".format(float(sum(self.sent_correct.values())) / sum(self.sent_length_totals.values()) * 100), "%"
		graph = Pyasciigraph()
		histogram = [(i, a/b) for (i,a),b in zip(self.sent_correct.iteritems(), self.sent_length_totals.itervalues())]
		for line in  graph.graph('% Correct by Sentence length', histogram):
			print(line)

	@classmethod
	def print_solution(cls, sentence, guess, gold):
		row_format = '{0}'
		for k, w in enumerate(sentence):
			row_format += "{"+str(k+1)+":<"+str(max(len(w), len(gold[k]))+1)+"}"
		print row_format.format("words: ", *sentence)
		print row_format.format("gold:  ", *gold)
		print row_format.format("guess: ", *guess)
