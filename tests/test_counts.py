import pytest
from os import path

from cyk.Counts import Counts
from cyk.pcfg import PCFG


HERE = path.dirname(__file__)


@pytest.fixture
def model():
	model = Counts()
	model.count_trees(path.join(HERE, 'bnc_test_trees.txt'), 1)
	return model

def test_normalisepings():
	assert Counts.normalise("90") == '!twoDigitNum'
	assert Counts.normalise("1990") == '!fourDigitNum'
	assert Counts.normalise("09-96") == '!containsDigitAndDash'
	assert Counts.normalise("R2D2") == '!containsDigitAndAlpha'
	assert Counts.normalise("11/9/00") == '!containsDigitAndSlash'
	assert Counts.normalise("23,000.00") == '!containsDigitAndComma'
	assert Counts.normalise("1.00") == '!containsDigitAndPeriod'
	assert Counts.normalise("1234567890") == '!otherNumber'
	assert Counts.normalise("ABC") == '!allCaps'
	assert Counts.normalise("A.") == '!capPeriod'
	assert Counts.normalise("Alexander") == '!initCap'
	assert Counts.normalise("can") == '!lowercase'
	assert Counts.normalise(".") == "!other"

def test_counting(model):

	assert model.get_pos_tags('to') == set([u'IN'])
	assert model.have_seen('Aberarder') is True
	assert model.have_seen('dslkjdfskjfdh') is False
	VPrule = model.get_binary_by_left_corner('VP')
	assert len(VPrule) == 3
	assert VPrule == ()

	print "Done"

def test_pcfg(model):
	PCFG(model)