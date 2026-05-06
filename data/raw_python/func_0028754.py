def _remap_new_key(self, indices, new_key, axis):
        """
        Return a key of type int, slice, or tuple that represents the
        combination of new_key with the given indices.

        Raises IndexError/TypeError for invalid keys.

        """
        size = len(indices)
        if _is_scalar(new_key):
            if new_key >= size or new_key < -size:
                msg = 'index {0} is out of bounds for axis {1}' \
                      ' with size {2}'.format(new_key, axis, size)
                raise IndexError(msg)
            result_key = indices[new_key]
        elif isinstance(new_key, slice):
            result_key = indices.__getitem__(new_key)
        elif isinstance(new_key, np.ndarray) and \
                new_key.dtype == np.dtype('bool'):
            # Numpy boolean indexing.
            if new_key.size > size:
                msg = 'too many boolean indices. Boolean index array ' \
                      'of size {0} is greater than axis {1} with ' \
                      'size {2}'.format(new_key.size, axis, size)
                raise IndexError(msg)
            result_key = tuple(np.array(indices)[new_key])
        elif isinstance(new_key, collections.Iterable) and \
                not isinstance(new_key, six.string_types):
            # Make sure we capture the values in case we've
            # been given a one-shot iterable, like a generator.
            new_key = tuple(new_key)
            for sub_key in new_key:
                if sub_key >= size or sub_key < -size:
                    msg = 'index {0} is out of bounds for axis {1}' \
                          ' with size {2}'.format(sub_key, axis, size)
                    raise IndexError(msg)
            result_key = tuple(indices[key] for key in new_key)
        else:
            raise TypeError('invalid key {!r}'.format(new_key))
        return result_key