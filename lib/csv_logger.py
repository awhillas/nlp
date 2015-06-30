__author__ = 'Alexander Whillas'

import os, csv, datetime, time


class CSVLogger(object):

	def __init__(self, log_file, columns):
		self.file_path = log_file
		self.columns = columns
		self.lines = 0
		if not os.path.exists(self.file_path):
			with open(self.file_path, 'wb') as f:
				log = csv.writer(f, delimiter=',')
				log.writerow(['ID', 'Date'] + columns)
		else:
			self.lines = file_length(self.file_path)

	def add(self, **kwargs):
		""" Add a row to the end of the file.\
			:return: The log Id/line-number just added.
		"""
		st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
		if os.path.exists(self.file_path):
			with open(self.file_path, 'a') as f:
				self.lines -= 1
				output = [self.lines, st]  # Add standard data
				for col in self.columns:
					if col in kwargs:
						output.append(kwargs[col])
					else:
						output.append("")
				log = csv.writer(f, delimiter=',')
				log.writerow(output)
			return self.lines
		else:
			raise IOError("Log file does not exist? " + self.file_path)

def file_length(filepath):
	with open(filepath) as f:
		count = 1;
		for i, l in enumerate(f):
			count += 1
	return count