#! /usr/bin/python

import sys, ConfigParser, importlib, time
from datetime import date

class Experiment:

	def __init__(self, data_set_id, scripts_pipe, config_path):
		self.data_set_id = data_set_id
		self.scripts_pipe = scripts_pipe
		self.config = ConfigParser.SafeConfigParser()
		self.config.read(config_path)
		today = date.fromtimestamp(time.time())
		self.working_dir = '/'.join([self.config.get(data_set_id, 'output'), today.isoformat(), self.data_set_id])

	def run(self):
		previous_module = None
		# Modules should communicate via the config object
		for script in self.scripts_pipe.split("|"):

			# Load the next script in the que
			m = self.load(script)

			# Load save previous module if not present
			if previous_module is None:
				if not m.input_module is None:
					previous_module = self.load(m.input_module)
					previous_module.load()
					print "Loaded: ", previous_module.__class__.__name__

			# Do the work
			success = m.run(previous_module)
			previous_module = m

			# Save the run to disk
			if success:
				m.save(self.working_dir)
			else:
				# Need to pull and error here
				print str(script) + " failed!"
				break

	def load(self, name):
		"""
		Dynamically load a module and instantiate its class.
		"""
		package, cls = name.rsplit('.', 1)
		module = importlib.import_module(name, package)
		m = getattr(module, cls)
		return m(self.config, self.data_set_id)


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print """
Too few arguments!

Run an experiment. General interface to machine learning scripts. The string parameter is a
pipeline of scripts that can be executed, with the output of one feed into the input of the next. You
can specify which computations you want to execute. Store the results of each part of the computation
to disk.

Example:
	The following command runs the preprocess_data, initialise_model and
train_model scripts, "preprocess_data|initialise_model|train_model" or run the only the train_model
script but also evaluates its performance "train_model|evaluate_model"

Usage:
	./run_exp.py "data_set_id" "script|pipe|string" ["path/to/config.ini"]
		"""
		sys.exit(1)
	elif len(sys.argv) <= 3:
		# Assume the config.ini file is in the same folder
		e = Experiment(sys.argv[1], sys.argv[2], "./config.ini")
	else:
		e = Experiment(sys.argv[1], sys.argv[2], sys.argv[3])
	e.run()