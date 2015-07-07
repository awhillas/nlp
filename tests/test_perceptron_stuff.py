__author__ = 'Alexander Whillas'
"""
"""

import pytest
from lib.PerceptronParser import format_data, deformat

@pytest.fixture
def words():
	return ['a', 'b', 'c', 'd']

@pytest.fixture
def tags():
	return ['X', 'Y', 'Z', 'W']

@pytest.fixture
def heads():
	return [None, 1,2, 0,3]


def test_format_data(words, tags, heads):
	w, t, h = format_data(words, tags, heads)

	assert w[0] == '<start>'
	assert w[1] == 'a'
	assert w[len(heads)] == 'ROOT'

	assert t[0] == '<start>'
	assert t[1] == 'X'
	assert t[len(heads)] == 'ROOT'

	assert h[0] is None
	assert h[1] == 1
	assert h[2] == 2
	assert h[3] == len(heads)
	assert h[4] == 3

def test_deformat(heads):
	h = deformat(heads)
	assert h[3] == 0