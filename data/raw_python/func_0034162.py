def from_pyvalue(name, value, **kwargs):
	"""
	Convenience wrapper for new_param() that constructs a Param element
	from an instance of a Python builtin type.  See new_param() for a
	description of the valid keyword arguments.
	"""
	return new_param(name, ligolwtypes.FromPyType[type(value)], value, **kwargs)