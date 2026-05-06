def map(self, func):
		"""
		Return a dictionary of the results of func applied to each
		of the segmentlist objects in self.

		Example:

		>>> x = segmentlistdict()
		>>> x["H1"] = segmentlist([segment(0, 10)])
		>>> x["H2"] = segmentlist([segment(5, 15)])
		>>> x.map(lambda l: 12 in l)
		{'H2': True, 'H1': False}
		"""
		return dict((key, func(value)) for key, value in self.iteritems())