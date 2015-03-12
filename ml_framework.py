
__author__ = "Alexander Whillas <whillas@gmail.com>"
__date__ = "$Jan 2015"


"""
	Provide generic interfaces to machine learning scripts.
	Taking design advice from: 	http://arkitus.com/patterns-for-research-in-machine-learning/
"""

import cPickle as pickle
import time
import os
from datetime import date

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
		self._config = config
		self._data_id = data_set_id
		print self.__class__

	def run(self, previous):
		""" Do the work
		"""
		return False  # return True if all went well
	
	def save(self, path):
		""" Save the output.
			Ideally so that an run() can be skipped 
		"""
		self.check_path(path)
		f = open(path+'/'+self.get_save_file_name(), 'wb')
		pickle.dump(self.__dict__, f, 2)
		f.close()

	def load(self):
		""" Load the saved output
			Instead of of run()?
		"""
		data = open(self.working_dir() + '/' + self.get_save_file_name(), 'rb')
		tmp_dict = pickle.load(data)
		data.close()
		self.__dict__.update(tmp_dict) 
		
	def get_save_file_name(self):
		""" Return a unique filename.
		"""
		return self.__class__.__name__ + "_data.pickle"

	def get_input_file_name(self):
		return self.config("training_file")

	def get_pickle_file(self):
		return self.working_dir() + '/' + self.get_save_file_name()

	def working_dir(self, check=True):
		today = date.fromtimestamp(time.time())
		path = '/'.join([self.config('working'), today.isoformat(), self._data_id])
		if check:
			self.check_path(path)
		return path

	def output_dir(self, check=True):
		today = date.fromtimestamp(time.time())
		path = '/'.join([self.config('output'), today.isoformat(), self._data_id])
		if check:
			self.check_path(path)
		return path

	def config(self, variable):
		""" Accessor for the config
		:param variable: section in the ini file
		:return: str
		"""
		return self._config.get(self._data_id, variable)

	@classmethod
	def check_path(cls, path):
		if not os.path.exists(path):
			os.makedirs(path)