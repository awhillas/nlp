__author__ = "Alexander Whillas <whillas@gmail.com>"

from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.measure import ConfusionMatrix
from lib.PerceptronTagger import PerceptronTagger


class Test(MachineLearningModule):

	def __init__(self, experiment):
		MachineLearningModule.__init__(self, experiment)
		self.input_module = 'me.Predict'

	def run(self, tagger):
		# TODO: move most of this inside the confusion matrix

		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		gold_labeled_sequences = data.tagged_sents(self.config('testing_file'))
		tron = PerceptronTagger(load=False)
		tron.load(loc=self.working_dir()+'/PerceptronTaggerModel.pickle')
		all_labels = tron.classes

		matrix = ConfusionMatrix(all_labels)
		word_count, word_error, sentence_errors = matrix.compare(tagger.labeled_sequences, gold_labeled_sequences)

		print "Words:", word_count, "Errors:", word_error
		print "Tag:", "{:>4.2f}".format(matrix.precision() * 100), "%"
		print "Sentence: ", "{:>4.2f}".format(float(sentence_errors) / len(gold_labeled_sequences) * 100), "%"
		
		# Save confusion matrix
		itr = int(self.config('iterations'))
		reg = float(self.config('regularization'))
		mxitr = int(self.config('maxiter'))
		m = len(gold_labeled_sequences)
		self.out("Perceptron-confusion_matrix,iter-{0},reg-{1},maxiter-{2},tag-{3:4.1f}%,sent-{4:4.1f}%.csv".format(itr, reg, mxitr, matrix.precision() * 100, float(sentence_errors) / m * 100), matrix.csv())
		
		matrix.show()

		return True
