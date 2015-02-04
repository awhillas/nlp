class Measure:
	""" Class for scoring a result.
		Has all the usual stuff.
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