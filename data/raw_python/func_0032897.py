def _partially_evaluate(self, addr, simplify=False):
        """
        Return part of the lazy array.
        """
        if self.is_homogeneous:
            if simplify:
                base_val = self.base_value
            else:
                base_val = self._homogeneous_array(addr) * self.base_value
        elif isinstance(self.base_value, (int, long, numpy.integer, float, bool)):
            base_val = self._homogeneous_array(addr) * self.base_value
        elif isinstance(self.base_value, numpy.ndarray):
            base_val = self.base_value[addr]
        elif have_scipy and sparse.issparse(self.base_value):  # For sparse matrices larr[2, :]
            base_val = self.base_value[addr]
        elif callable(self.base_value):
            indices = self._array_indices(addr)
            base_val = self.base_value(*indices)
            if isinstance(base_val, numpy.ndarray) and base_val.shape == (1,):
                base_val = base_val[0]
        elif hasattr(self.base_value, "lazily_evaluate"):
            base_val = self.base_value.lazily_evaluate(addr, shape=self._shape)
        elif isinstance(self.base_value, VectorizedIterable):
            partial_shape = self._partial_shape(addr)
            if partial_shape:
                n = reduce(operator.mul, partial_shape)
            else:
                n = 1
            base_val = self.base_value.next(n)  # note that the array contents will depend on the order of access to elements
            if n == 1:
                base_val = base_val[0]
            elif partial_shape and base_val.shape != partial_shape:
                base_val = base_val.reshape(partial_shape)
        elif isinstance(self.base_value, collections.Iterator):
            raise NotImplementedError("coming soon...")
        else:
            raise ValueError("invalid base value for array (%s)" % self.base_value)
        return self._apply_operations(base_val, addr, simplify=simplify)