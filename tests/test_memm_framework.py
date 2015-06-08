__author__ = 'Alexander Whillas'
"""
Test all the bits around the MEMM but not the actual MEMM itself
"""

import pytest
from lib.MaxEntMarkovModel import *

@pytest.mark.parametrize("input,expected", [
	# Things with digits
	('90', '!twoDigitNum!'),
	('1990', '!fourDigitNum!'),
	('09-96', '!containsDigitAndDash!'),
	('R2D2', '!containsDigitAndAlpha!'),
	('11/9/00', '!containsDigitAndSlash!'),
	('23,000.00', '!containsDigitAndComma!'),
	('1.00', '!containsDigitAndPeriod!'),
	('1234567890', '!otherNum!'),
	(# Email regex
	'me@yomumas.com', "!email!"),
	('me@yomumas4ever.com', "!email!"),
	('me2@yomumas4ever.com', "!email!"),
	('McConnell@ECT', '!email!'),
	(# URL regex
	'http://www.quora.com/', "!url!"),
	('www.quora.com', "!url!"),
	('zombo.com', "!url!"),
	('zombo.com.au', "!url!"),
	('http://www.quora.com/#123', "!url!"),
	(# Junk
	'---------', '!allPunctuation!'),
	('!@#$%^&*', '!allPunctuation!'),
	('!!!!', '!allPunctuation!'),

	('U.S.A', '!acronym!'),
	('U.S.A.', '!acronym!'),
	('A.', '!initial!'),
	('bong.', '!abbreviation!'),

	# Emoticons
	(':)', '!emoticon!'),
	(';-)', '!emoticon!'),

	# Pass thought
	('-', '-'),
	('.', '.'),
	(',', ','),
	(';', ';'),
	('-LRB-', '-lrb-'),
])
def test_normalize_numbers_and_junk(input, expected):
	assert CollinsNormalisation.word(input) == expected

@pytest.mark.parametrize("input,expected", [
	('ABC', '!allCaps!'),
	('A.', '!capPeriod!'),
	('Alexander', '!initCap!'),
	('can', '!lowercase!'),
])
def test_rare_words(input, expected):
	assert CollinsNormalisation.low_freq_words(input) == expected


@pytest.fixture
def labeled_sequence():
	return (['a', 'b', 'something', 'd', 'e'], ['1', '2', 'middle', '4', '5'])

def test_Ratnaparkhi96Features(labeled_sequence):
	features = Ratnaparkhi96Features.get(2, labeled_sequence)

	print features

	#assert 'bias' in features  # Acts like a prior

	assert 'i word something' in features
	assert 'i-1 word b' in features
	assert 'i-2 word a' in features
	assert 'i+1 word d' in features
	assert 'i+2 word e' in features

	# Bigram's
	assert 'i-2 word, i-1 word a b' in features
	assert 'i-1 word, i word b something' in features
	assert 'i word, i+1 word something d' in features
	assert 'i+1 word, i+2 word d e' in features

	# Bigram tags (skip-grams? in features
	assert 'i-1 tag 2' in features
	assert 'i-2 tag 1' in features
	assert 'i+1 tag 4' in features
	assert 'i+2 tag 5' in features

	# Tri-gram tags
	assert 'i-2 tag, i-1 tag 1 2' in features
	assert 'i-1 tag, i+1 tag 2 4' in features
	assert 'i+1 tag, i+2 tag 4 5' in features

	assert 'i word suffix g' in features
	assert 'i word suffix ng' in features
	assert 'i word suffix ing' in features
	assert 'i word suffix hing' in features

def test_SequenceFeaturesTemplate_on_long_word():
	suffixes = SequenceFeaturesTemplate.get_suffixes('running', 4)
	assert 'g' in suffixes
	assert 'ng' in suffixes
	assert 'ing' in suffixes
	assert 'ning' in suffixes

def test_SequenceFeaturesTemplate_on_short_word():
	suffixes = SequenceFeaturesTemplate.get_suffixes('at', 4)
	assert 't' in suffixes

def test_context_is_pseudo():
	assert Context.is_pseudo('!pseudo!')
	assert Context.is_pseudo('!pseudo!!pseudo!')

def test_context_get_features_general(labeled_sequence):
	c = Context(labeled_sequence, feature_templates=Ratnaparkhi96Features)
	features = c.get_features(2, 'test')

	print features
	#assert 'bias' in features  # Acts like a prior

	assert 'i word something &test' in features
	assert 'i-1 word b &test' in features
	assert 'i-2 word a &test' in features
	assert 'i+1 word d &test' in features
	assert 'i+2 word e &test' in features

	# Bigram's
	assert 'i-2 word, i-1 word a b &test' in features
	assert 'i-1 word, i word b something &test' in features
	assert 'i word, i+1 word something d &test' in features
	assert 'i+1 word, i+2 word d e &test' in features

	# Bigram tags (skip-grams? in features
	assert 'i-1 tag 2 &test' in features
	assert 'i-2 tag 1 &test' in features
	assert 'i+1 tag 4 &test' in features
	assert 'i+2 tag 5 &test' in features

	# Tri-gram tags
	assert 'i-2 tag, i-1 tag 1 2 &test' in features
	assert 'i-1 tag, i+1 tag 2 4 &test' in features
	assert 'i+1 tag, i+2 tag 4 5 &test' in features

	assert 'i word suffix g &test' in features
	assert 'i word suffix ng &test' in features
	assert 'i word suffix ing &test' in features
	assert 'i word suffix hing &test' in features

def test_context_get_features_edge_case(labeled_sequence):
	c = Context(labeled_sequence, feature_templates=Ratnaparkhi96Features)
	features = c.get_features(0, 'test')
	BEGIN = Context.BEGIN_SYMBOL

	assert 'i-1 word {0} &test'.format(BEGIN) in features
	assert 'i-2 word {0} &test'.format(BEGIN*2) in features

	# Bigram's
	assert 'i-2 word, i-1 word {0} {1} &test'.format(BEGIN*2, BEGIN) in features
	assert 'i-1 word, i word {0} a &test'.format(BEGIN) in features

	# Bigram tags (skip-grams? in features
	assert 'i-1 tag {0} &test'.format(BEGIN) in features
	assert 'i-2 tag {0} &test'.format(BEGIN*2) in features

	# Tri-gram tags
	assert 'i-2 tag, i-1 tag {0} {1} &test'.format(BEGIN*2,BEGIN) in features
	assert 'i-1 tag, i+1 tag {0} 2 &test'.format(BEGIN) in features

def test_merge_features():
	values = [1,2,3]
	keys = ['a', 'b', 'c']
	out = MaxEntMarkovModel._merge_weight_values(keys, values)
	assert out['a'] == 1
	assert out['b'] == 2
	assert out['c'] == 3