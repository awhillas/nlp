#! /usr/bin/python

__author__="Alexander Whillas <whillas@gmail.com>"
__date__ ="$Jan 2015"

import sys, ConfigParser, importlib, os
from datetime import date

class Experiment:
	
	def __inti__(self, data_set_id, scripts_pipe, config_path):
		self.config = ConfigParser.SafeConfigParser()
		self.config.read(config_path)
		self.working_dir = '/'.join([config.read(data_set_id, 'output'), date.isoformat(), data_set_id])
	
	def run(self):
		previous_module = None
		# Modules should communicate via the config object 
		for script in self.scripts_pipe.split("|"):
			# Load the next script in the que
			module = importlib.import_module(script)
			m = module(config, data_set_id)
			# Load save previous module if not present
			if previous_module is None:
				if not m.input_module is None:
					previous_module = importlib.import_module(m.input_module)
					previous_module.load(working_dir)
			# Do the work
			success = m.run(previous_module)
			previous_module = m
			# Save the run to disk
			if success:
				if not os.path.exists(working_dir):
					os.makedirs(working_dir)
				m.save(working_dir)
			else:
				# Need to pull and error here
				print str(script) + " failed!"
				break

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print """
Too few arguments!

Run an experiment. General interface to machine learning scripts. The string paramater is a
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