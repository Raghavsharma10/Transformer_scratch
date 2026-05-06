def copy(self, keys = None):
		"""
		Return a copy of the segmentlistdict object.  The return
		value is a new object with a new offsets attribute, with
		references to the original keys, and shallow copies of the
		segment lists.  Modifications made to the offset dictionary
		or segmentlists in the object returned by this method will
		not affect the original, but without using much memory
		until such modifications are made.  If the optional keys
		argument is not None, then should be an iterable of keys
		and only those segmentlists will be copied (KeyError is
		raised if any of those keys are not in the
		segmentlistdict).

		More details.  There are two "built-in" ways to create a
		copy of a segmentlist object.  The first is to initialize a
		new object from an existing one with

		>>> old = segmentlistdict()
		>>> new = segmentlistdict(old)

		This creates a copy of the dictionary, but not of its
		contents.  That is, this creates new with references to the
		segmentlists in old, therefore changes to the segmentlists
		in either new or old are reflected in both.  The second
		method is

		>>> new = old.copy()

		This creates a copy of the dictionary and of the
		segmentlists, but with references to the segment objects in
		the original segmentlists.  Since segments are immutable,
		this effectively creates a completely independent working
		copy but without the memory cost of a full duplication of
		the data.
		"""
		if keys is None:
			keys = self
		new = self.__class__()
		for key in keys:
			new[key] = _shallowcopy(self[key])
			dict.__setitem__(new.offsets, key, self.offsets[key])
		return new