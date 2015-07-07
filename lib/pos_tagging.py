"""
Utilities for POS tagging
"""

import time, datetime
from lib.filelock import FileLock
from os import path
import cPickle as pickle


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
	start = time.time()
	total_sents = len(sequence_list)
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
		print "Time:", '%.3f' % (t1 - t0), ", Per word:", '%.3f' % ((t1 - t0) / len(unlabeled_sequence))
		print "Estimated time:", datetime.timedelta(seconds=(t1 - start) / i * (total_sents - i)), "\n"
		out += [zip(unlabeled_sequence, tags)]

	return out


def tag_all_shared(sequence_list, tagger, normaliser=None, working_path='', block_size=10, output_pickle='shared.pickle'):
	""" Uses file locking to shared the tagging process amongst multiple machines that share a common file system.
	"""
	out = {}
	total_sents = len(sequence_list)
	counter_file = working_path+'/_tagger_position_counter.txt'  # share where we're up to in the sequence_list
	log_file = working_path + output_pickle + '.log'
	start = 0
	while start != -1:
		with FileLock(counter_file):  # lock semaphore
			if path.exists(counter_file):
				with open(counter_file, 'r') as f:
					start = int(f.readline())

			if start == -1:
				break

			if start + block_size < total_sents:  # process another block
				new_start = stop = start + block_size
			elif start + block_size >= total_sents:  # last block
				new_start = -1
				stop = total_sents

			with open(counter_file, 'w') as f:
				f.write(str(new_start))

			safe_log(log_file, '{0} sentences, doing {1} to {2}\n'.format(total_sents, start, stop))

		for i in xrange(start, stop):
			print "Sentence {0} ({1:2.2f}%)".format(i, float(i)/total_sents * 100)
			seq = sequence_list[i]
			display = [seq]

			t0 = time.time()

			if normaliser is not None:
				normalised_seq = normaliser.sentence(seq)
				display += [normalised_seq]
				tags = tagger(normalised_seq)
			else:
				tags = tagger(seq)

			display += [tags]
			t1 = time.time()

			print matrix_to_string(display)
			print "Time:", '%.3f' % (t1 - t0), ", Per word:", '%.3f' % ((t1 - t0) / len(seq))
			out["".join(seq)] = tags
	
	safe_log(log_file, "{0} Saving tagged examples\n".format(len(out)))
	update_shared_dict(out, working_path + output_pickle)  # finished so write the output to a common pickled dict

def safe_log(log_file, text):
	with FileLock(log_file):
		with open(log_file, 'a') as f:
			f.write(text)
	

def update_shared_dict(data, filepath):
	shared_data = {}
	with FileLock(filepath):  # Lock semaphore

		if path.exists(filepath):
			shared_data.update(pickle.load(open(filepath)))

		shared_data.update(data)

		pickle.dump(shared_data, open(filepath, 'wb'), -1)

	return True
