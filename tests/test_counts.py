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

	# Check Look up rule with VP in left hand corner
	VP_rule = model.get_binary_by_left_corner('VP')
	assert len(VP_rule) == 1
	(X, Y, Z) = VP_rule.copy().pop()
	assert (Y, Z) in model.N[X].keys()

	# other way around: look up all S rules and then check their left-hand-corner's
	for (Y, Z), _ in model.N['S'].iteritems():
		count = 0
		for rule in model.get_binary_by_left_corner(Y):
			if rule == (u'S', Y, Z):
				count += 1
		assert count == 1, "Missing rule in get_binary_by_left_corner counts: " + str((u'S', Y, Z)) + " in: " + str(model.reverseN_left_hand_corner[Y])

	print "Done"

def test_pcfg(model):
	pcfg = PCFG(model)

	pos = pcfg.get_word_pos_tags('the')  # for a known word.
	assert 'DT' in pos
	pos = pcfg.get_word_pos_tags('VP')  # for a unary.
	assert pos == []
	pos = pcfg.get_word_pos_tags('fdhskjlhfsl')  # for an unknown word.
	assert 'DT' in pos

	unary_rules = pcfg.get_unary_rules_for(['VP'])

	rules = pcfg.lookup_rules_for(list(pos), list(pos))
	print rules