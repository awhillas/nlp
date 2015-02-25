
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
		return self.config.get(self.data_id, "training_file")

	def get_output_file_name(self):
		return self.working_dir() + '/' + self.get_save_file_name()

	def working_dir(self, check=True):
		today = date.fromtimestamp(time.time())
		path = '/'.join([self.config.get(self.data_id, 'working'), today.isoformat(), self.data_id])
		if check:
			self.check_path(path)
		return path

	def output_dir(self, check=True):
		today = date.fromtimestamp(time.time())
		path = '/'.join([self.config.get(self.data_id, 'output'), today.isoformat(), self.data_id])
		if check:
			self.check_path(path)
		return path

	@classmethod
	def check_path(cls, path):
		if not os.path.exists(path):
			os.makedirs(path)