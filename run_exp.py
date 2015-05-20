#! /usr/bin/python

import os, sys, ConfigParser, importlib, time, argparse
from datetime import date

# Fix for picking instance methods

def _pickle_method(method):
	func_name = method.im_func.__name__
	obj = method.im_self
	cls = method.im_class
	return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
	try:
		func = cls.__dict__[func_name]
	except KeyError:
		pass
	return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class Experiment:

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
		self.output = {
			"Name:": args.name,
			"Comment:": args.comment
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
		path = '/'.join([self.config(name), self.get_date(), self._data_set_id, self.name])
		if check:
			self.check_path(path)
		return path

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
	def check_path(cls, path):
		if not os.path.exists(path):
			os.makedirs(path)

	def load(self, name):
		"""
		Dynamically load a module and instantiate its class.
		"""
		package, cls = name.rsplit('.', 1)
		module = importlib.import_module(name, package)
		m = getattr(module, cls)
		return m(self)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="""
		Run an experiment. General interface to machine learning scripts. The string parameter is a
		pipeline of scripts that can be executed, with the output of one feed into the input of the next. You
		can specify which computations you want to execute. Store the results of each part of the computation
		to disk.
	""")
	parser.add_argument('data',\
						help="Section ID in the config file that points to the experiment variables.")
	parser.add_argument('-n', '--name', default="",\
						help="Name used to label the output and working sets.")
	parser.add_argument('-m', '--comment', default="",\
						help="Comment to be logged with the output.")
	parser.add_argument('script', nargs='+',\
						help="Script pipe, an ordered list of scripts to run.")
	parser.add_argument('-c', '--config', default='./config.ini',\
						help='Path to a config file to use. If not provided "config.ini" in the same folder is assumed.')
	parser.add_argument('-nc', '--no-cache', default=False, action='store_const', const=True,\
						help='Do not load saved modules from file.')
	parser.add_argument('-ns', '--no-save', default=False, action='store_const', const=True,\
						help='Do not save modules to disk after a run.')

	args = parser.parse_args()

	if args.data and len(args.script) > 0:
		e = Experiment(args)
		e.run()