import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""
		Run an experiment. General interface to machine learning scripts. The string parameter is a
		pipeline of scripts that can be executed, with the output of one feed into the input of the next. You
		can specify which computations you want to execute. Store the results of each part of the computation
		to disk.
	""")
	parser.add_argument('data',\
						help="Section ID in the config file that points to the experiment variables.")
	parser.add_argument('-n', '--name', default="",
						help="Name used to label the output and working sets.")
	parser.add_argument('-m', '--comment', default="",
						help="Comment to be logged with the output.")
	parser.add_argument('script', nargs='+',
						help="Script pipe, an ordered list of scripts to run.")

	parser.add_argument('-r', '--regularization', default=0.33, type=float,
						help="MEMM's regularization constant.")
	parser.add_argument('-i', '--maxiter', default=1, type=int,
						help="Maximum iterations for the MEMM's param. optimisation.")
	parser.add_argument('-a', '--ambiguity', default=0.1, type=float,
						help="Multi-tagging ambiguity level. The lower the closer to zero is more ambiguous.")

	parser.add_argument('-c', '--config', default='./config.ini',
						help='Path to a config file to use. If not provided "config.ini" in the same folder is assumed.')
	parser.add_argument('-nc', '--no-cache', default=False, action='store_const', const=True,
						help='Do not load saved modules from file.')
	parser.add_argument('-ns', '--no-save', default=False, action='store_const', const=True,
						help='Do not save modules to disk after a run.')
	parser.add_argument('-nl', '--no-log', default=False, action='store_const', const=True,
						help='Do not log the results of the experiment.')

	args = parser.parse_args()

	if args.data and len(args.script) > 0:
		exp = Experiment(args)