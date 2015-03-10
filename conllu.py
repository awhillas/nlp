# CoNLL-U Format Reader for the Natural Language Toolkit
# http://universaldependencies.github.io/docs/format.html
#
# Based on the Natural Language Toolkit: Dependency Corpus Reader
#
# Author: Alexander Whillas <whillas@gmail.com>
#
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

import codecs

from nltk.parse import DependencyGraph
from nltk.tokenize import *

from nltk.data import PathPointer

from nltk.corpus.reader.util import StreamBackedCorpusView, concat, read_blankline_block
from nltk.corpus.reader.api import SyntaxCorpusReader, CorpusReader

class ConlluReader(SyntaxCorpusReader):

	def __init__(self, root, fileids, encoding='utf8',
				 word_tokenizer=TabTokenizer(),
				 sent_tokenizer=RegexpTokenizer('\n', gaps=True),
				 para_block_reader=read_blankline_block):

		CorpusReader.__init__(self, root, fileids, encoding)

	#########################################################

	def raw(self, fileids=None):
		"""
		:return: the given file(s) as a single string.
		:rtype: str
		"""
		result = []
		for fileid, encoding in self.abspaths(fileids, include_encoding=True):
			if isinstance(fileid, PathPointer):
				result.append(fileid.open(encoding=encoding).read())
			else:
				with codecs.open(fileid, "r", encoding) as fp:
					result.append(fp.read())
		return concat(result)

	def words(self, fileids=None):
		return concat([ConlluView(fileid, False, False, False, encoding=enc)
					   for fileid, enc in self.abspaths(fileids, include_encoding=True)])

	def tagged_words(self, fileids=None):
		return concat([ConlluView(fileid, True, False, False, encoding=enc)
					   for fileid, enc in self.abspaths(fileids, include_encoding=True)])

	def sents(self, fileids=None):
		return concat([ConlluView(fileid, False, True, False, encoding=enc)
					   for fileid, enc in self.abspaths(fileids, include_encoding=True)])

	def tagged_sents(self, fileids=None):
			return concat([ConlluView(fileid, True, True, False, encoding=enc)
						   for fileid, enc in self.abspaths(fileids, include_encoding=True)])

	def parsed_sents(self, fileids=None):
		sents=concat([ConlluView(fileid, False, True, True, encoding=enc)
					  for fileid, enc in self.abspaths(fileids, include_encoding=True)])
		return [DependencyGraph(sent) for sent in sents]

	def sents_len_counts(self, fileids=None):
		""" Average sentence length
		:param fileids: file IDs
		:return: dict of sentence length counts
		"""
		lengths = ()



class ConlluView(StreamBackedCorpusView):
	def __init__(self, corpus_file, tagged, group_by_sent, dependencies,
				 chunk_types=None, encoding='utf8'):
		self._tagged = tagged
		self._dependencies = dependencies
		self._group_by_sent = group_by_sent
		self._chunk_types = chunk_types
		StreamBackedCorpusView.__init__(self, corpus_file, encoding=encoding)

	def read_block(self, stream):
		# Read the next sentence.
		sent = read_blankline_block(stream)[0].strip()

		# extract word and tag from any of the formats
		if not self._dependencies:
			lines = [line.split('\t') for line in sent.split('\n')]
			if len(lines[0]) == 10:
				sent = [(line[1], line[4]) for line in lines]
			else:
				raise ValueError('Unexpected number of fields in dependency tree file: %d', len(lines[0]))

			# discard tags if they weren't requested
			if not self._tagged:
				sent = [word for (word, tag) in sent]

		# Return the result.
		if self._group_by_sent:
			return [sent]
		else:
			return list(sent)
