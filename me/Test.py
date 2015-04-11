class Test(MachineLearningModule):

	def __init__(self, config, data_set_id):
		MachineLearningModule.__init__(self, config, data_set_id)
		self.input_module = 'me.Tag'

	def run(self, Tagger):
		predicted_label = Tagger.labled_sequences
