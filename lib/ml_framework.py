__author__ = "Alexander Whillas <whillas@gmail.com>"


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

	def __init__(self, experiment):
		"""
		:param config: Instance of ConfigParser.
		:param data_set_id: data set ID which should be a group in the .ini file
		"""
		self.input_module = None
		self._experiment = experiment
		self._config = experiment.config
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
			path = self._experiment.dir('working')
		full_path = path + '/' + self.get_save_file_name(filename_prefix)
		f = open(full_path, 'wb')
		pickle.dump(self.__dict__, f, 2)
		f.close()
		print "Saved", full_path
		return full_path

	def load(self, path = None, filename_prefix = ''):
		""" Load the saved output
			Instead of of run()?
		"""
		if path is None:
			path  = self.working_dir()
		full_path = path + '/' + self.get_save_file_name(filename_prefix)
		if not os.path.exists(full_path):
			print "Could not load", full_path
			return False
		data = open(full_path, 'rb')
		tmp_dict = pickle.load(data)
		data.close()
		self.__dict__.update(tmp_dict)
		print "Loaded", full_path
		return full_path

	def get_save_file_name(self, filename_prefix = ''):
		""" Return a unique filename.
		"""
		return self.__class__.__name__ + "_data" + filename_prefix + ".pickle"

	def get_input_file_name(self):
		return self.config("training_file")

	def pickle_file(self):
		return self.working_dir() + '/' + self.get_save_file_name()

	def working_dir(self):
		return self._experiment.dir('working')

	def output_dir(self):
		return self._experiment.dir('output')

	def config(self, variable):
		""" Accessor for the config
		:param variable: section in the ini file
		:return: str
		"""
		return self._experiment.config(variable)