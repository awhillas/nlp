__author__ = "Alexander Whillas <whillas@gmail.com>"

from sortedcontainers import SortedDict
import re
import csv
import StringIO

class Measure:
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

class ConfusionMatrix:
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

