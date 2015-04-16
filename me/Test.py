class Test(MachineLearningModule):

	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'me.Tag'

	def run(self, tagger):
		predicted = tagger.labeled_sequences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		gold = data.tagged_sents(self.config('testing_file'))

