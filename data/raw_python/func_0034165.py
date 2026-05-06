def yaml_to_param(obj, name):
	"""
	Return the top-level element of a document sub-tree containing the
	YAML serialization of a Python object.
	"""
	return from_pyvalue(u"yaml:%s" % name, unicode(yaml.dump(obj)))