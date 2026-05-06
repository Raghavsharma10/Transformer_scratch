def method2pos(method):
	''' Returns a list of valid POS-tags for a given method. '''

	if method in ('articles', 'plural', 'miniaturize', 'gender'):
		pos = ['NN']
	elif method in ('conjugate',):
		pos = ['VB']
	elif method in ('comparative, superlative'):
		pos = ['JJ']
	else:
		pos = ['*']
	return pos