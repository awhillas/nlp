__author__ = 'Alexander Whillas'
"""
"""

import pytest
from lib.PerceptronParser import get_stack_context_2D, get_buffer_context_2D, get_parse_context_2D

@pytest.fixture
def tags():
	return [
		None,
		{'a':0.1, 'b':0.2, 'c':0.3},#1
		{'d':0.4, 'e':0.5},			#2
		{'f':0.6},					#3
		{'h':0.7, 'i':0.8},			#4
		{'j':0.9},					#5
	]

def test_get_stack_context_2D(tags):
	returned = get_stack_context_2D(1, [3], tags)
	assert 'f' in returned

	returned = get_stack_context_2D(2, [3, 5], tags)
	assert 'f' in returned
	assert 'j' in returned

	returned = get_stack_context_2D(3, [1, 2, 3], tags)
	assert 'a' in returned
	assert 'b' in returned
	assert 'c' in returned
	assert 'd' in returned
	assert 'e' in returned
	assert 'f' in returned

	returned = get_stack_context_2D(0, [], tags)
	assert len(returned) == 0


def test_get_buffer_context_2D(tags):
	r1 = get_buffer_context_2D(5, len(tags), tags)
	assert len(r1) == 1
	assert not 'h' in r1
	assert 'j' in r1

	r2 = get_buffer_context_2D(4, len(tags), tags)
	assert len(r2) == 3
	assert not 'f' in r1
	assert 'h' in r2
	assert 'i' in r2
	assert 'j' in r2

	r3 = get_buffer_context_2D(3, len(tags), tags)
	assert len(r3) == 4
	assert not 'd' in r1
	assert 'f' in r3
	assert 'h' in r3
	assert 'i' in r3
	assert 'j' in r3

	r4 = get_buffer_context_2D(1, len(tags), tags)
	print r4
	assert len(r4) == 6
	assert 'a' in r4
	assert 'b' in r4
	assert 'c' in r4
	assert 'd' in r4
	assert 'e' in r4
	assert 'f' in r4
	assert not 'h' in r1


def test_get_parse_context_2D(tags):
	# r1 = get_parse_context_2D(-1, [], tags)
	# assert not r1
	#
	# r2 = get_parse_context_2D(0, [[]], tags)
	# assert not r2

	r3 = get_parse_context_2D(0, [[1, 3, 5]], tags)
	assert len(r3) == 2
	assert 'j' in r3
	assert 'f' in r3

	r4 = get_parse_context_2D(0, [[5]], tags)
	assert len(r4) == 1
	assert 'j' in r4

	r2 = get_parse_context_2D(1, [[1,2,3], [5]], tags)
	assert len(r2) == 1
	assert 'j' in r2
