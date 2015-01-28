
__author__ = "Alexander Whillas <whillas@gmail.com>"
__date__ = "$Jan 2015"


"""
	Provide generic interfaces to machine learning scripts.
	Taking design advice from: 	http://arkitus.com/patterns-for-research-in-machine-learning/
"""

import cPickle as pickle


class MachineLearningModule:  # Interface.
	""" Generic interface to all modules in an ML chain/pipeline.
		The general idea is that parts of the pipeline can be run interchangeably
		or simply loaded from a previous run in case of crashes in the middle of long
		pipelines or if it is redundant to keep recalculating the same thing.
	"""
	
	def __init__(self, config, data_set_id):
		"""
		:param config: Instance of ConfigParser.
		:param data_set_id: data set ID which should be a group in the .ini file
		"""
		self.input_module = None
		self.config = config
		self.data_id = data_set_id
		print self.__class__

	def run(self, previous):
		""" Do the work
		"""
		return False  # return True if all went well
	
	def save(self, path):
		""" Save the output.
			Ideally so that an run() can be skipped 
		"""
		f = open(path+'/'+self.get_save_file_name(), 'wb')
		pickle.dump(self.__dict__, f, 2)
		f.close()
	
	def load(self, path):
		""" Load the saved output
			Instead of of run()?
		"""
		data = open(path + '/' + self.get_save_file_name(), 'rb')
		tmp_dict = pickle.load(data)
		data.close()
		self.__dict__.update(tmp_dict) 
		
	def get_save_file_name(self):
		""" Return a unique filename.
		"""
		return self.__class__.__name__ + ".pickle.data"