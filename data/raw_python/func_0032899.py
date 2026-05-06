def evaluate(self, simplify=False, empty_val=0):
        """
        Return the lazy array as a real NumPy array.

        If the array is homogeneous and ``simplify`` is ``True``, return a
        single numerical value.
        """
        # need to catch the situation where a generator-based larray is evaluated a second time
        if self.is_homogeneous:
            if simplify:
                x = self.base_value
            else:
                x = self.base_value * numpy.ones(self._shape, dtype=self.dtype)
        elif isinstance(self.base_value, (int, long, numpy.integer, float, bool, numpy.bool_)):
            x = self.base_value * numpy.ones(self._shape, dtype=self.dtype)
        elif isinstance(self.base_value, numpy.ndarray):
            x = self.base_value
        elif callable(self.base_value):
            x = numpy.array(numpy.fromfunction(self.base_value, shape=self._shape, dtype=int), dtype=self.dtype)
        elif hasattr(self.base_value, "lazily_evaluate"):
            x = self.base_value.lazily_evaluate(shape=self._shape)
        elif isinstance(self.base_value, VectorizedIterable):
            x = self.base_value.next(self.size)
            if x.shape != self._shape:
                x = x.reshape(self._shape)
        elif have_scipy and sparse.issparse(self.base_value):  # For sparse matrices
            if empty_val!=0:
                x = self.base_value.toarray((sparse.csc_matrix))
                x = numpy.where(x, x, numpy.nan)
            else:
                x = self.base_value.toarray((sparse.csc_matrix))
        elif isinstance(self.base_value, collections.Iterator):
            x = numpy.fromiter(self.base_value, dtype=self.dtype or float, count=self.size)
            if x.shape != self._shape:
                x = x.reshape(self._shape)
        else:
            raise ValueError("invalid base value for array")
        return self._apply_operations(x, simplify=simplify)