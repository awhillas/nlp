__author__ = 'Alexander Whillas'

import yaml

"""
Wishlist:
- want to be able to apply filters to data i.e. is_projective, normalization etc
- save and load just the models
""


class Experiment(object):
	""" Manage:
		- loading the config
		- running the Tasks
		- ranging over all parameter combo's
		- saving & loading the model
	"""
	def __init__(self, config_yaml):
		with open(config_yaml) as f:
			self.config = yaml.safe_load(f)

	def run(self, data, task):

	def generate_parameters(self, task):
		"""
		:return: List of parameter dictionaries
		"""


class Task(object):



class model(object):
	def __init__(self, model):
		self._model = model
	
	def train(self):
		self._model.train()
		
	def predict(self, method):
		self._model.method()
		pass
	
	def test(self):
		pass