def value(self, indexes, int_as_index=False):
        """
        Wrapper function for get. It will return a list, no index. If the indexes are integers it will be assumed
        that they are locations unless int_as_index = True. If the indexes are locations then they will be rotated to 
        the left by offset number of locations.

        :param indexes: integer location, single index, list of indexes or list of boolean
        :param int_as_index: if True then will treat int index values as indexes and not locations
        :return: value or list of values
        """
        # single integer value
        if isinstance(indexes, int):
            if int_as_index:
                return self.get(indexes, as_list=True)
            else:
                indexes = indexes - self._offset
                return self._data[indexes]

        # slice
        elif isinstance(indexes, slice):
            if isinstance(indexes.start, int) and not int_as_index:  # treat as location
                start = indexes.start - self._offset
                stop = indexes.stop - self._offset + 1  # to capture the last value
                # check locations are valid and will not return empty
                if start > stop:
                    raise IndexError('end of slice is before start of slice')
                if (start > 0 > stop) or (start < 0 < stop):
                    raise IndexError('slide indexes invalid with given offset:%f' % self._offset)
                # where end is the last element
                if (start < 0) and stop == 0:
                    return self._data[start:]
                return self._data[start:stop]
            else:  # treat as index
                indexes = self._slice_index(indexes)
                return self.get(indexes, as_list=True)

        # list of booleans
        elif all([isinstance(x, bool) for x in indexes]):
            return self.get(indexes, as_list=True)

        # list of values
        elif isinstance(indexes, list):
            if int_as_index or not isinstance(indexes[0], int):
                return self.get(indexes, as_list=True)
            else:
                indexes = [x - self._offset for x in indexes]
                return self.get_locations(indexes, as_list=True)

        # just a single value
        else:
            return self.get(indexes)