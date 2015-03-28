import pytest
from lib.distance import *

def test_Levenshtein():
	""" Levenshtein distance test
	"""
	assert optimalAlignmentDistance('123abc', '123abc') == 0
	assert optimalAlignmentDistance('abort', 'abortt') == 1, "Inerts are not counted correctly"
	assert optimalAlignmentDistance('abortt', 'abort') == 1, "Deletes not counted correctly"
	assert optimalAlignmentDistance('bag', 'bog') == 1, "Replacements not counted correctly"
	assert optimalAlignmentDistance('best', 'ebst'), "Transposition not counted correctly"
	assert optimalAlignmentDistance('which', 'witch') == 2, "Combo insert and replace"