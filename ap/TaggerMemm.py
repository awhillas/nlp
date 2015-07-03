from lib.ml_framework import MachineLearningModule
from lib.conllu import ConlluReader
from lib.MaxEntMarkovModel import MaxEntMarkovModel, Ratnaparkhi96Features, CollinsNormalisation

class TaggerMemm(MachineLearningModule):
	""" Train MEMM tagger.
	"""
	def run(self, _=None):

		# Get (words, tags) sequences for all sentences
		data = ConlluReader(self.config('uni_dep_base'), '.*\.conllu')  # Corpus
		training = data.tagged_sents(self.config('training_file'))

		print "Training MaxEnt tagger"

		reg = float(self.config('regularization'))
		mxitr = int(self.config('maxiter'))
		ambiguity = self.log_me('Ambiguity', float(self.config('ambiguity')))

		# 10 fold cross validation
		self.tagged = {}
		num_folds = 10
		subset_size = len(training)/num_folds
		for i in range(num_folds):
			testing = training[i*subset_size:][:subset_size]
			learning = training[:i*subset_size] + training[(i+1)*subset_size:]
			self.tagger.train(learning, regularization=reg, maxiter=mxitr, optimise=False)
			print "Generating Muti POS Tags"
			word_count = 0; tag_count = 0
			for sentence in testing:
				word_count += len(sentence)
				multi_tags = self.tagger.multi_tag(sentence, ambiguity)
				tags = [max(all_tags.iterkeys(), key=(lambda key: all_tags[key])) for all_tags in multi_tags]
				tag_count += sum([len(tags) for tags in tags])
				self.tagged["".join(sentence)] = (tags, multi_tags)  # save for the parser as it needs real tagger output.
				print 'Tagged: {0} of {1}'.format(i+1, len(testing))
			print "Tags per word %.3f" % self.log_me('Tags / word', round(float(tag_count) / word_count, 2))
		return True

	def save(self, path = None, filename_prefix = ''):
		ambiguity = float(self.config('ambiguity'))
		self.backup(self.tagged, self.working_dir() + 'memm_multi_tagged_sentences_ambiguity-%.2f.pickle' % ambiguity)
		return self.tagger.save(self.working_dir())

	def load(self, path = None, filename_prefix = ''):
		ambiguity = float(self.config('ambiguity'))
		# self.tagged = self.restore(self.tagged, self.working_dir() + 'memm_multi_tagged_sentences_ambiguity-%.2f.pickle' % ambiguity)
		self.tagger = MaxEntMarkovModel(feature_templates=Ratnaparkhi96Features, word_normaliser=CollinsNormalisation)
		return self.tagger.load(self.working_dir(), filename_prefix)