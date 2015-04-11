
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
		raise NotImplementedError( "Should have implemented this" )
		return False  # return True if all went well
	
	def save(self, path = None, filename_prefix = ''):
		""" Save the output.
			Ideally so that a run() can be skipped
		"""
		if path is None:
			path = self.working_dir()
		self.check_path(path)
		f = open(path+'/'+self.get_save_file_name(filename_prefix), 'wb')
		pickle.dump(self.__dict__, f, 2)
		f.close()

	def load(self, path = None, filename_prefix = ''):
		""" Load the saved output
			Instead of of run()?
		"""
		if path is None:
			path  = self.working_dir()
		data = open(path+ '/' + self.get_save_file_name(filename_prefix), 'rb')
		tmp_dict = pickle.load(data)
		data.close()
		self.__dict__.update(tmp_dict) 
		
	def get_save_file_name(self, filename_prefix = ''):
		""" Return a unique filename.
		"""
		return self.__class__.__name__ + "_data" + filename_prefix + ".pickle"

	def get_input_file_name(self):
		return self.config("training_file")

	def pickle_file(self):
		return self.working_dir() + '/' + self.get_save_file_name()

	def working_dir(self, check=True):
		today = date.fromtimestamp(time.time())
		path = '/'.join([self.config('working'), today.isoformat(), self._data_id])
		if check:
			self.check_path(path)
		return path

	def output_dir(self, check=True):
		path = '/'.join([self.config('output'), self.get_date(), self._data_id])
		if check:
			self.check_path(path)
		return path

	def get_date(self):
		today = date.fromtimestamp(time.time())
		return today.isoformat()

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