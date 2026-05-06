def contains(self, other):
		"""
		Returns True if offset vector @other can be found in @self,
		False otherwise.  An offset vector is "found in" another
		offset vector if the latter contains all of the former's
		instruments and the relative offsets among those
		instruments are equal (the absolute offsets need not be).

		Example:

		>>> a = offsetvector({"H1": 10, "L1": 20, "V1": 30})
		>>> b = offsetvector({"H1": 20, "V1": 40})
		>>> a.contains(b)
		True

		Note the distinction between this and the "in" operator:

		>>> "H1" in a
		True
		"""
		return offsetvector((key, offset) for key, offset in self.items() if key in other).deltas == other.deltas