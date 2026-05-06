def check_bounds(self, addr):
        """
        Check whether the given address is within the array bounds.
        """
        def is_boolean_array(arr):
            return hasattr(arr, 'dtype') and arr.dtype == bool

        def check_axis(x, size):
            if isinstance(x, (int, long, numpy.integer)):
                lower = upper = x
            elif isinstance(x, slice):
                lower = x.start or 0
                upper = min(x.stop or size - 1, size - 1)  # slices are allowed to go past the bounds
            elif isinstance(x, collections.Sized):
                if is_boolean_array(x):
                    lower = 0
                    upper = x.size - 1
                else:
                    if len(x) == 0:
                        raise ValueError("Empty address component (address was %s)" % str(addr))
                    if hasattr(x, "min"):
                        lower = x.min()
                    else:
                        lower = min(x)
                    if hasattr(x, "max"):
                        upper = x.max()
                    else:
                        upper = max(x)
            else:
                raise TypeError("Invalid array address: %s (element of type %s)" % (str(addr), type(x)))
            if (lower < -size) or (upper >= size):
                raise IndexError("Index out of bounds")
        full_addr = self._full_address(addr)
        if isinstance(addr, numpy.ndarray) and addr.dtype == bool:
            if len(addr.shape) > len(self._shape):
                raise IndexError("Too many indices for array")
            for xmax, size in zip(addr.shape, self._shape):
                upper = xmax - 1
                if upper >= size:
                    raise IndexError("Index out of bounds")
        else:
            for i, size in zip(full_addr, self._shape):
                check_axis(i, size)