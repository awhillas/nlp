__author__ = 'alex'

from lib.PerceptronParser import DefaultList

def dep_tree_to_list(deptree):
	# Oh, for an abstraction to the data...!
	words = DefaultList('')
	tags = DefaultList('')
	heads = [None]
	# labels = [None]
	for i, n in deptree.nodes.iteritems():
		if n['word'] is not None:
			words.append(n['word'])  # TODO: perhaps intern(n['word'].encode('utf8')) would speed things up but this need to be consistent and do it in the tagger too :-?220
			tags.append(n['tag'])    #  same here!
			heads.append(n['head'])
			# heads.append(n['head'] if n['head'] != 0 else len(deptree.nodes))  # ROOT moved to the end
			# labels.append(n['rel'])
	return words, tags, heads #, labels

def is_projective(heads):
	length = len(heads)
	# heads = [n['head'] for n in dep_tree.nodes.itervalues()]
	for w1 in range(length):
		if heads[w1] is not None:
			h1 = heads[w1]
			for w2 in range(length):
				if heads[w2] is not None and arcs_cross(w1, h1, w2, heads[w2]):
					return False
	return True

def arcs_cross(w1, h1, w2, h2):
	if w1 > h1:
		w1, h1 = h1, w1
	if w2 > h2:
		w2, h2 = h2, w2
	if w1 > w2:
		w1, h1, w2, h2 = w2, h2, w1, h1
	return w1 < w2 < h1 < h2 or w1 < w2 == h2 < h1
