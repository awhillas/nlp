
__author__="Alexander Whillas <whillas@gmail.com>"
__date__ ="$Jan 2015"


"""
	Provide generic interfaces to machine learning scripts.
	Takign design advice from: 	http://arkitus.com/patterns-for-research-in-machine-learning/
"""

import cPickle as pickle
import importlib

class MachineLearningModule:	# Interface.
	""" Generic interface to all moudles in an ML lifecycle.
		The generial idea is that parts of the pipeline can be run interchangabley 
		or simply loaded from a previous run incase of crashes in the middle of long
		pipelines or if it is redundant to keep recalculating the same thing.
	"""
	input_module = None
	
	def __init__(self, config, data_set_id):		
		self.config = config 
		self.data_id = data_set_id
		self.inti()
	
	def inti():
		""" Overwrite me in a subclass to initialise self
		"""
		pass
			
	def run(self, input):
		""" Do the work
		"""
		return False	# return True is=f all went well
	
	def save(self, path):
		""" Save the output.
			Ideally so that an run() can be skipped 
		"""
		f = open(path+'/'+self.getSaveFileName(), 'wb')
		pickle.dump(self.__dict__,f,2)
		f.close()
	
	def load(self, path):
		""" Load the saved output
			Instead of of run()?
		"""
		file = open(path+'/'self.getSaveFileName(),'rb')
		tmp_dict = pickle.load(file)
		file.close()
		self.__dict__.update(tmp_dict) 
		
	def getSaveFileName():
		""" Return a unique filename.
		"""
		return type(self).__name__ + ".pickle.data"