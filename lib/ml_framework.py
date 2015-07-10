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
import gzip

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
		self.no_log = args.no_log
		self.name = args.name
		self.comment = args.comment

		self.params = {}
		self.params.update(vars(args))

		# Dict. we can throw output into
		self._log = {
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
			m.load(self.dir('working'))

			# Load save previous module if not present
			if previous_module is None and not self.no_cache:
				if not m.input_module is None:
					previous_module = self.load(m.input_module)
					previous_module.load(self.dir('working'))
					print "Loaded:", previous_module.__class__.__name__

			# Do the work
			success = m.run(previous_module)
			previous_module = m

			# Save the run to disk
			if success:
				if not self.no_save:
					print "Saving", script
					m.save(self, self.dir('working'))
			else:
				# Need to pull and error here
				print str(script) + " failed! Work not saved."
				break

	def dir(self, name, check=True):
		dir = path.join(self.get(name), self._data_set_id, self.name)
		if check:
			self.check_path(dir)
		return dir

	def out(self, file_name, text):
		with open(self.dir('output')+'/'+file_name, 'a') as f:
			f.write(text)

	def get(self, var):
		if var in self.params:
			out = self.params[var]
			self._log.update({var: out})
			print var, out
		else:
			out = self._config.get(self._data_set_id, var)
		return out

	def log(self, what, value):
		if not self.no_log:
			self._log.update({what: value})

	@classmethod
	def get_date(cls):
		today = date.fromtimestamp(time.time())
		return today.isoformat()

	@classmethod
	def check_path(cls, dir):
		""" Check to see if a path exists and if not then create it. """
		if not path.exists(dir):
			os.makedirs(dir)

	def load(self, name):
		""" Dynamically load a module and instantiate its class."""
		package, cls = name.rsplit('.', 1)
		module = importlib.import_module(name, package)
		m = getattr(module, cls)
		new = m(self)
		return new


class MachineLearningModule:  # Interface.
	""" Generic interface to all modules in an ML chain/pipeline.
		The general idea is that parts of the pipeline can be run interchangeably
		or simply loaded from a previous run in case of crashes in the middle of long
		pipelines or if it is redundant to keep recalculating the same thing.
	"""
	PREVIOUS_MODULE = None

	def __init__(self, experiment):
		self._experiment = experiment
		self.keepers = {}  # what to keep and pass on down the pipe. All that gets saved to disk.
		self.input_module = self.PREVIOUS_MODULE  # Expected model to run before this one. This is clumsy.

	def run(self, previous):
		""" Do the work
		"""
		raise NotImplementedError( "Should have implemented this" )
		return False  # return True if all went well

	def save(self, data, path = None):
		""" Save the output.
			Ideally so that a run() can be skipped
		"""
		full_path = self.get_save_file_name(path)
		copy = dict(self.__dict__)
		copy.pop('_experiment')  # don't save
		self.backup(copy, full_path)
		return full_path

	def load(self, path = None, filename_prefix = ''):
		""" Load the saved output
			Instead of of run()?
		"""
		full_path = self.get_save_file_name(path)
		tmp_dict = self.restore(full_path)
		if tmp_dict:
			tmp_dict['_experiment'] = self._experiment
			self.__dict__.update(tmp_dict)
			print "Loaded", full_path
			return full_path
		else:
			return False

	@classmethod
	def backup(cls, data, save_path):
		""" Save the given data. """
		with gzip.open(save_path+".gz", 'wb') as f:
			pickle.dump(data, f, 2)
			print "Saved", save_path

	@classmethod
	def restore(cls, save_dir):
		""" Load the given file and return it. """
		save_dir_gz = save_dir+".gz"
		if not path.exists(save_dir_gz):
			print "Could not load", save_dir
			return None
		else:
			with gzip.open(save_dir_gz, 'rb') as f:
				return pickle.load(f)

	def delete(self, path = None, filename_prefix = ''):
		""" Remove saved file
		"""
		full_path = self.get_save_file_name(path, filename_prefix)
		if os.path.exists(full_path):
			print "Removing", full_path
			os.remove(full_path)

	def get_save_file_name(self, path = None):
		""" Return a unique filename.
		"""
		if path is None:
			path = self.dir('working')
		return path + '/' + self.__class__.__name__ + "_data.pickle"

	def dir(self, name):
		return self._experiment.dir(name)

	def get_input_file_name(self):
		return self.get("training_file")

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
		print "MachineLearningModule.config() DEPRECIATED"
		return self._experiment.get(variable)

	def get(self, variable):
		""" Accessor for the config
		:param variable: section in the ini file
		:return: str
		"""
		return self._experiment.get(variable)

	def get_log(self):
		return self._experiment.log

	def out(self, file_name, text):
		self._experiment.out(file_name, text)

	def log(self, key, value):
		self._experiment.log(key, value)
		return value

	def cols(self):
		return self._experiment.log.keys()