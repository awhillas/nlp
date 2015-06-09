# from: http://stackoverflow.com/questions/10379948/defaultlist-design
__author__ = 'alex'

class DefaultList(list):
	def __init__(self, default_factory, *args, **kwargs):
		#super(list).__init__(*args, **kwargs)
		list.__init__(self, *args, **kwargs)
		self.default_factory = default_factory

	def __getitem__(self, key):
		try:
			return super(list).__getitem__(key)
		except IndexError:
			return self.default_factory()

	def __setitem__(self, key, value):
		try:
			super(list).__setitem__(key, value)
		except IndexError:
			for i in range(len(self), key):
				self.append(self.default_factory())
			self.append(value)