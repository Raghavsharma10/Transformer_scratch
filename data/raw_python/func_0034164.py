def pickle_from_param(elem, name):
	"""
	Retrieve a pickled Python object from the document tree rooted at
	elem.
	"""
	return pickle.loads(str(get_pyvalue(elem, u"pickle:%s" % name)))