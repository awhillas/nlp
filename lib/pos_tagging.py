"""
Utilities for POS tagging
"""

import time

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

def tag_all(sequence_list, tagger, normaliser=None, output_file=None):
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
		display = [unlabeled_sequence]

		t0 = time.time()

		if normaliser is not None:
			normalised_seq = normaliser.sentence(unlabeled_sequence)
			display += [normalised_seq]
			tags = tagger(normalised_seq)
		else:
			tags = tagger(unlabeled_sequence)

		display += [tags]
		t1 = time.time()

		print matrix_to_string(display)
		print "Time:", '%.3f' % (t1 - t0), ", Per word:", '%.3f' % ((t1 - t0) / len(unlabeled_sequence)), "\n"
		out += [zip(unlabeled_sequence, tags)]

	return out
