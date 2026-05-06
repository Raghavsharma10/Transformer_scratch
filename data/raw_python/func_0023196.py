def insert(self, index, data, itemsize=None):
        """ Insert data before index

        Parameters
        ----------

        index : int
            Index before which data will be inserted.

        data : array_like
            An array, any object exposing the array interface, an object
            whose __array__ method returns an array, or any (nested) sequence.

        itemsize:  int or 1-D array
            If `itemsize is an integer, N, the array will be divided
            into elements of size N. If such partition is not possible,
            an error is raised.

            If `itemsize` is 1-D array, the array will be divided into
            elements whose succesive sizes will be picked from itemsize.
            If the sum of itemsize values is different from array size,
            an error is raised.
        """

        if not self._sizeable:
            raise AttributeError("List is not sizeable")

        if isinstance(data, (list, tuple)) and isinstance(data[0], (list, tuple)):  # noqa
            itemsize = [len(l) for l in data]
            data = [item for sublist in data for item in sublist]

        data = np.array(data, copy=False).ravel()
        size = data.size

        # Check item size and get item number
        if itemsize is not None:
            if isinstance(itemsize, int):
                if (size % itemsize) != 0:
                    raise ValueError("Cannot partition data as requested")
                _count = size // itemsize
                _itemsize = np.ones(_count, dtype=int) * (size // _count)
            else:
                _itemsize = np.array(itemsize, copy=False)
                _count = len(itemsize)
                if _itemsize.sum() != size:
                    raise ValueError("Cannot partition data as requested")
        else:
            _count = 1

        # Check if data array is big enough and resize it if necessary
        if self._size + size >= self._data.size:
            capacity = int(2 ** np.ceil(np.log2(self._size + size)))
            self._data = np.resize(self._data, capacity)

        # Check if item array is big enough and resize it if necessary
        if self._count + _count >= len(self._items):
            capacity = int(2 ** np.ceil(np.log2(self._count + _count)))
            self._items = np.resize(self._items, (capacity, 2))

        # Check index
        if index < 0:
            index += len(self)
        if index < 0 or index > len(self):
            raise IndexError("List insertion index out of range")

        # Inserting
        if index < self._count:
            istart = index
            dstart = self._items[istart][0]
            dstop = self._items[istart][1]
            # Move data
            Z = self._data[dstart:self._size]
            self._data[dstart + size:self._size + size] = Z
            # Update moved items
            I = self._items[istart:self._count] + size
            self._items[istart + _count:self._count + _count] = I

        # Appending
        else:
            dstart = self._size
            istart = self._count

        # Only one item (faster)
        if _count == 1:
            # Store data
            self._data[dstart:dstart + size] = data
            self._size += size
            # Store data location (= item)
            self._items[istart][0] = dstart
            self._items[istart][1] = dstart + size
            self._count += 1

        # Several items
        else:
            # Store data
            dstop = dstart + size
            self._data[dstart:dstop] = data
            self._size += size

            # Store items
            items = np.ones((_count, 2), int) * dstart
            C = _itemsize.cumsum()
            items[1:, 0] += C[:-1]
            items[0:, 1] += C
            istop = istart + _count
            self._items[istart:istop] = items
            self._count += _count