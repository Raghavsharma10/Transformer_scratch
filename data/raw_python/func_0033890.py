def from_array(name, array, dim_names = None):
	"""
	Construct a LIGO Light Weight XML Array document subtree from a
	numpy array object.

	Example:

	>>> import numpy, sys
	>>> a = numpy.arange(12, dtype = "double")
	>>> a.shape = (4, 3)
	>>> from_array(u"test", a).write(sys.stdout)	# doctest: +NORMALIZE_WHITESPACE
	<Array Type="real_8" Name="test:array">
		<Dim>3</Dim>
		<Dim>4</Dim>
		<Stream Delimiter=" " Type="Local">
			0 3 6 9
			1 4 7 10
			2 5 8 11
		</Stream>
	</Array>
	"""
	# Type must be set for .__init__();  easier to set Name afterwards
	# to take advantage of encoding handled by attribute proxy
	doc = Array(Attributes({u"Type": ligolwtypes.FromNumPyType[str(array.dtype)]}))
	doc.Name = name
	for n, dim in enumerate(reversed(array.shape)):
		child = ligolw.Dim()
		if dim_names is not None:
			child.Name = dim_names[n]
		child.pcdata = unicode(dim)
		doc.appendChild(child)
	child = ArrayStream(Attributes({u"Type": ArrayStream.Type.default, u"Delimiter": ArrayStream.Delimiter.default}))
	doc.appendChild(child)
	doc.array = array
	return doc