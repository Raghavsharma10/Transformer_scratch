def pickle_to_param(obj, name):
	"""
	Return the top-level element of a document sub-tree containing the
	pickled serialization of a Python object.
	"""
	return from_pyvalue(u"pickle:%s" % name, unicode(pickle.dumps(obj)))