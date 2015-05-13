#! /usr/bin/python

import sys, ConfigParser, importlib, time, argparse
from datetime import date

class Experiment:

	def __init__(self, data_set_id, scripts_pipe, config_file, no_cache, no_save):
		self.data_set_id = data_set_id
		self.scripts_pipe = scripts_pipe
		self.config = ConfigParser.SafeConfigParser()
		self.config.read(config_file)
		today = date.fromtimestamp(time.time())
		self.working_dir = '/'.join([self.config.get(data_set_id, 'working'), today.isoformat(), self.data_set_id])
		self.no_cache = no_cache
		self.no_save = no_save

	def run(self):
		previous_module = None
		# Modules should communicate via the config object
		for script in self.scripts_pipe:

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
					m.save(self.working_dir)
			else:
				# Need to pull and error here
				print str(script) + " failed! Work not saved."
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
	parser = argparse.ArgumentParser(description="""
		Run an experiment. General interface to machine learning scripts. The string parameter is a
		pipeline of scripts that can be executed, with the output of one feed into the input of the next. You
		can specify which computations you want to execute. Store the results of each part of the computation
		to disk.
	""")
	parser.add_argument('data',\
						help="Section ID in the config file that points to the experiment variables.")
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
		e = Experiment(args.data, args.script, args.config, args.no_cache, args.no_save)
		e.run()