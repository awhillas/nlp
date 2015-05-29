__author__ = "Alexander Whillas <whillas@gmail.com>"


"""
	Provide generic interfaces to machine learning scripts.
	Taking design advice from: 	http://arkitus.com/patterns-for-research-in-machine-learning/
"""

import cPickle as pickle
import os

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

	def run(self, previous):
		""" Do the work
		"""
		raise NotImplementedError( "Should have implemented this" )
		return False  # return True if all went well

	def save(self, path = None, filename_prefix = ''):
		""" Save the output.
			Ideally so that a run() can be skipped
		"""
		full_path = self.get_save_file_name(path, filename_prefix)
		copy = dict(self.__dict__)
		copy.pop('_experiment')  # don't save
		with open(full_path, 'wb') as f:
			pickle.dump(copy, f, 2)
		print "Saved", full_path
		return full_path

	def load(self, path = None, filename_prefix = ''):
		""" Load the saved output
			Instead of of run()?
		"""
		full_path = self.get_save_file_name(path, filename_prefix)
		if not os.path.exists(full_path):
			print "Could not load", full_path
			return False
		with open(full_path, 'rb') as f:
			tmp_dict = pickle.load(f)
		tmp_dict['_experiment'] = self._experiment
		self.__dict__.update(tmp_dict)
		print "Loaded", full_path
		return full_path

	def delete(self, path = None, filename_prefix = ''):
		""" Remove saved file
		"""
		full_path = self.get_save_file_name(path, filename_prefix)
		if os.path.exists(full_path):
			print "Removing", full_path
			os.remove(full_path)

	def get_save_file_name(self, path = None, filename_prefix = ''):
		""" Return a unique filename.
		"""
		if path is None:
			path = self._experiment.dir('working')
		return path + self.__class__.__name__ + "_data" + filename_prefix + ".pickle"

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
		
	def out(self, file_name, text):
		self._experiment.out(file_name, text)
