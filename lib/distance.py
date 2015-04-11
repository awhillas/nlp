"""
	Distance metric functions.
"""

def optimalAlignmentDistance(str1, str2):
	""" Levenshtein distance with one additional recurrence
		see: https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
	"""
	lenStr1 = len(str1)
	lenStr2 = len(str2)

	# d is a table with lenStr1+1 rows and lenStr2+1 columns
	d = [[0 for j in xrange(1, lenStr1)] for i in xrange(1, lenStr2)]
 
	# for loop is inclusive, need table 1 row/column larger than string length
	for i in xrange(0, lenStr1):
		d[i, 0] = i
	for j in xrange(1, lenStr2):
		d[0, j] = j
 
	# pseudo-code assumes string indices start at 1, not 0
	# if implemented, make sure to start comparing at 1st letter of strings
	for i in xrange(1, lenStr1):
		for j in xrange(1, lenStr2):
			if str1[i] == str2[j]:
				cost = 0
			else:
				cost = 1
			d[i, j] = min(
				d[i-1, j] + 1,  # deletion
				d[i, j-1] + 1,  # insertion
				d[i-1, j-1] + cost  # substitution
			)
			if i > 1 and j > 1 and str1[i] == str2[j-1] and str1[i-1] == str2[j]:
				d[i, j] = min(d[i, j], d[i-2, j-2] + cost)  # transposition
	return d[lenStr1, lenStr2]