__author__ = "Alexander Whillas <whillas@gmail.com>"

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.measure import ConfusionMatrix
from lib.csv_logger import CSVLogger


class PosTest(MachineLearningModule):

	PREVIOUS_MODULE = 'dis.MemmTag'

	def run(self, previous):
		def print_sol(sentence, guess, gold):
			row_format = '{0}'
			for k, w in enumerate(words):
				row_format += "{"+str(k+1)+":<"+str(max(len(w), len(gold[k]))+1)+"}"
			print row_format.format("\nwords: ", *sentence)
			print row_format.format("gold:  ", *gold)
			print row_format.format("guess: ", *guess)

		predicted = previous.labeled_sequences
		data = ConlluReader(self.get('uni_dep_base'), '.*\.conllu')  # Corpus
		gold_labeled_sequences = data.tagged_sents(self.get('cv_file'))

		all_labels = previous.tagger.tag_count.keys()
		self.log('Features', "{:,}".format(len(previous.tagger.learnt_features.keys())))

		matrix = ConfusionMatrix(all_labels)
		sents = 0
		for i, gold_seq in enumerate(gold_labeled_sequences):
			words, gold_labels = zip(*gold_seq)
			predicted_labels = predicted[i]
			sentence_error = False
			if len(words) == len(predicted_labels):
				for j, word in enumerate(words):
					matrix.add(gold_labels[j], predicted_labels[j])
					if gold_labels[j] != predicted_labels[j]:
						sentence_error = True
				if not sentence_error:
					sents += 1
			else:
				raise Exception("Sequences out of sync '%s' and (%s) ", " ".join(words), " ".join(predicted_labels))
			print_sol(words, predicted_labels, gold_labels)
			error_count = sum([1 if predicted_labels[i] == gold_labels[i] else 0 for i,_ in enumerate(gold_labels)])
			print "Correct:", error_count, "/", len(words), ", rate:", "%.1f" % (float(error_count) / len(words) * 100), "%"

		print "Tag:", self.log("Word %", "{:>4.2f}".format(matrix.precision() * 100)), "%"
		print "Sentence: ", self.log("Sent. %", "{:>4.2f}".format(float(sents) / len(gold_labeled_sequences) * 100)), "%"

		if not self._experiment.no_log:
			columns = ['Name','Data','regularization','maxiter','Features','Word %','Sent. %','Total Time','Comment']
			logger = CSVLogger(self.dir('output') + "/pos-tagging.log.csv", columns)
			run_id = logger.add(**self._experiment._log)
			self.out("{:}_confusion_matrix,reg-{:0.4}.csv".format(run_id, self.get('regularization')), matrix.csv())

		return False
