# -*- coding: utf-8 -*-
__author__ = "Alexander Whillas <whillas@gmail.com>"


"""
	Provide generic interfaces to machine learning scripts.
	Taking design advice from: 	http://arkitus.com/patterns-for-research-in-machine-learning/
"""

import cPickle as pickle
import os
import os.path as path
import ConfigParser, importlib, time
# import copy_reg
# import types
from datetime import date


class Experiment(object):
	"""
	Experiment is a script pipe and some parameters to apply to this run of the scripts.
	Should handle the setup and ranging through parameter combinations
	"""
	def __init__(self, args):
		self._data_set_id = args.data
		self._scripts_pipe = args.script

		self._config = ConfigParser.SafeConfigParser()
		self._config.read(args.config)
		self._config.read(args.config)

		self.no_cache = args.no_cache
		self.no_save = args.no_save
		self.name = args.name
		self.comment = args.comment

		# Dict. we can throw output into
		self.log = {
			"Data": self._data_set_id,
			"Name": args.name,
			"Comment": args.comment,
		}

	def run(self):
		previous_module = None
		# Modules should communicate via the config object
		for script in self._scripts_pipe:

			# Load the next script in the que
			m = self.load(script)

			# Load save previous module if not present
			if previous_module is None and not self.no_cache:
				if not m.input_module is None:
					previous_module = self.load(m.input_module)
					previous_module.load()
					print "Loaded: ", previous_module.__class__.__name__

			# Do the work
			success = m.run(previous_module)
			previous_module = m

			# Save the run to disk
			if success:
				if not self.no_save:
					print "Saving", script
					m.save(self.dir('working'))
			else:
				# Need to pull and error here
				print str(script) + " failed! Work not saved."
				break

	def dir(self, name, check=True):
		dir = path.join(self.config(name), self._data_set_id, self.name)
		if check:
			self.check_path(dir)
		return dir

	def out(self, file_name, text):
		with open(self.dir('output')+'/'+file_name, 'a') as f:
			f.write(text)

	def config(self, variable):
		""" Accessor for the config
		:param variable: section in the ini file
		:return: str
		"""
		return self._config.get(self._data_set_id, variable)

	@classmethod
	def get_date(cls):
		today = date.fromtimestamp(time.time())
		return today.isoformat()

	@classmethod
	def check_path(cls, dir):
		""" Check to see if a path exists and if not then create it """
		if not path.exists(dir):
			os.makedirs(dir)

	def load(self, name):
		"""
		Dynamically load a module and instantiate its class.
		"""
		package, cls = name.rsplit('.', 1)
		module = importlib.import_module(name, package)
		m = getattr(module, cls)
		return m(self)


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
			path = self.dir('working')
		return path + '/' + self.__class__.__name__ + "_data" + filename_prefix + ".pickle"

	def dir(self, name):
		return self._experiment.dir(name)

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

	def log(self, file_name, text):
		self._experiment.log(file_name, text)


# Fix for picking instance methods

# def _pickle_method(method):
# 	func_name = method.im_func.__name__
# 	obj = method.im_self
# 	cls = method.im_class
# 	return _unpickle_method, (func_name, obj, cls)
#
# def _unpickle_method(func_name, obj, cls):
# 	try:
# 		func = cls.__dict__[func_name]
# 	except KeyError:
# 		pass
# 	return func.__get__(obj, cls)
#
# copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

