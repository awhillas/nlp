__author__ = "Alexander Whillas <whillas@gmail.com>"

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.measure import ConfusionMatrix
from lib.csv_logger import CSVLogger


class Test(MachineLearningModule):

	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.input_module = 'me.Predict'
		# Output logging
		log_keys = ["Name", "Date", "Model", "Data", "Features", "Train Iter.", "Reg.", "Max Iter.", "Word %", "Sentence %", "Comment"]
		self.logger = self.logger = CSVLogger(self.config('output') + "/POS-tagging.log.csv", log_keys)

	def run(self, tagger):
		# TODO: move most of this inside the confusion matrix
		def print_sol(sentence, guess, gold):
			row_format = '{0}'
			for k, w in enumerate(words):
				row_format += "{"+str(k+1)+":<"+str(max(len(w), len(gold[k]))+1)+"}"
			print row_format.format("\nwords: ", *sentence)
			print row_format.format("gold:  ", *gold)
			print row_format.format("guess: ", *guess)

		predicted = tagger.labeled_sequences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		gold_labeled_sequences = data.tagged_sents(self.config('testing_file'))

		all_labels = tagger.model.tag_count.keys()

		matrix = ConfusionMatrix(all_labels)
		sents = 0
		for i, gold_seq in enumerate(gold_labeled_sequences):
			words, gold_labels = zip(*gold_seq)
			words2, predicted_labels = zip(*predicted[i])
			sentence_error = False
			for j, word in enumerate(words):
				if word == words2[j]:
					matrix.add(gold_labels[j], predicted_labels[j])
					if gold_labels[j] != predicted_labels[j]:
						sentence_error = True
				else:
					print "Sequences out of sync", words, words2
					raise
			if not sentence_error:
				sents += 1
			print_sol(words, predicted_labels, gold_labels)
			error_count = sum([1 if predicted_labels[i] == gold_labels[i] else 0 for i,_ in enumerate(gold_labels)])
			print "Correct:", error_count, "/", len(words), ", rate:", "%.1f" % (float(error_count) / len(words) * 100), "%"

		print "Tag:", "{:>4.2f}".format(matrix.precision() * 100), "%"
		print "Sentence: ", "{:>4.2f}".format(float(sents) / len(gold_labeled_sequences) * 100), "%"

		if not self._experiment.no_save:
			itr = int(self.config('iterations'))
			reg = float(self.config('regularization'))
			mxitr = int(self.config('maxiter'))
			word_precision = matrix.precision() * 100
			sent_precision = float(sents) / len(gold_labeled_sequences) * 100
			# Save confusion matrix
			self.out("confusion_matrix,iter-{0},reg-{1},maxiter-{2},tag-{3:4.1f}%,sent-{4:4.1f}%.csv".format(itr, reg, mxitr, word_precision, sent_precision), matrix.csv())
			# Log run
			self.logger.add(**dict({
				"Date": self._experiment.get_date(),
				"Features": len(tagger.model.weights),
				"Model": "MaxEnt",
				"Data": self._experiment._data_set_id,
				"Train Iter.": itr,
				"Reg.": reg,
				"Max Iter.": mxitr,
				"Word %": round(word_precision, 2),
				"Sentence %": round(sent_precision, 2)
			}.items() + self._experiment.log.items()))

		return True
