from nltk.tree import Tree

class MyTree(Tree):
	@classmethod
	def from_list(cls, lst):
		if len(lst) > 1:
			if isinstance(lst[1], list):   # has children
				if len(lst) == 3:
					return Tree(lst[0], [cls.from_list(lst[1]), cls.from_list(lst[2])])
				elif len(lst) == 2:
					return Tree(lst[0], [cls.from_list(lst[1])])
			else:
				return Tree(lst[0], [lst[1]]) # just a list of children
		else:
			print("Not a binary tree?")
			print lst

	@classmethod
	def to_dict(cls, tree):
		tdict = {}
		for t in tree:
			if isinstance(t, Tree) and isinstance(t[0], Tree):
				tdict[t.node] = cls.to_dict(t)
			elif isinstance(t, Tree):
				tdict[t.node] = t[0]
		return tdict

	@classmethod
	def to_json(cls, dict):
		return json.dumps(cls.to_dict())