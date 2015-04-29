
from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.measure import ConfusionMatrix

class Test(MachineLearningModule):

	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'me.Predict'

	def run(self, tagger):
		predicted = tagger.labeled_sequences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		gold_labeled_sequences = data.tagged_sents(self.config('testing_file'))

		all_labels = tagger.model.tag_count.keys()
		matrix = ConfusionMatrix(all_labels)

		for i, gold_seq in enumerate(gold_labeled_sequences):
			words, gold_labels = zip(*gold_seq)
			words2, predicted_labels = predicted[i]
			for j, word in enumerate(words):
				if word == words2[j]:
					matrix.add(gold_labels[j], predicted_labels[j])
				else:
					print "Sequences out of sync", words, words2
					raise

		matrix.show(len(all_labels) * 3)
		print "Precision:", matrix.precision() * 100, "%"
