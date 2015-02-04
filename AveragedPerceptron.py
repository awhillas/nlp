

class AveragedPerceptron():
	""" From: Hal Daumé III (2012) 'A Course in Machine Learning, Chap.3 - Averaged perceptron'
	"""
	def __init__(self):
		self.weights =

	def max_iterations(self):
		""" Should be: 1/margin^2
		"""
		return 5

	def train(self, data, max_iter):
		for input, actual_class in data:
			predicted_class = self.predict(input)
			if actual_class != predicted_class:
				# Only update weights on wrong predictions

		pass

	def predict(self, features):
		return None

def _get_features(self, i, word, context, prev, prev2):
	""" Map tokens-in-contexts into a feature representation, implemented as a
		set. If the features change, a new model must be trained.
		Taken from: https://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
	"""
	def add(name, *args):
		features.add('+'.join((name,) + tuple(args)))

	features = set()
	add('bias') # This acts sort of like a prior
	add('i suffix', word[-3:])
	add('i pref1', word[0])
	add('i-1 tag', prev)
	add('i-2 tag', prev2)
	add('i tag+i-2 tag', prev, prev2)
	add('i word', context[i])
	add('i-1 tag+i word', prev, context[i])
	add('i-1 word', context[i-1])
	add('i-1 suffix', context[i-1][-3:])
	add('i-2 word', context[i-2])
	add('i+1 word', context[i+1])
	add('i+1 suffix', context[i+1][-3:])
	add('i+2 word', context[i+2])
	return features