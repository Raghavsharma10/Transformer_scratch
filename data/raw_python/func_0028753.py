def _cleanup_new_key(self, key, size, axis):
        """
        Return a key of type int, slice, or tuple that is guaranteed
        to be valid for the given dimension size.

        Raises IndexError/TypeError for invalid keys.

        """
        if _is_scalar(key):
            if key >= size or key < -size:
                msg = 'index {0} is out of bounds for axis {1} with' \
                      ' size {2}'.format(key, axis, size)
                raise IndexError(msg)
        elif isinstance(key, slice):
            pass
        elif isinstance(key, np.ndarray) and key.dtype == np.dtype('bool'):
            if key.size > size:
                msg = 'too many boolean indices. Boolean index array ' \
                      'of size {0} is greater than axis {1} with ' \
                      'size {2}'.format(key.size, axis, size)
                raise IndexError(msg)
        elif isinstance(key, collections.Iterable) and \
                not isinstance(key, six.string_types):
            # Make sure we capture the values in case we've
            # been given a one-shot iterable, like a generator.
            key = tuple(key)
            for sub_key in key:
                if sub_key >= size or sub_key < -size:
                    msg = 'index {0} is out of bounds for axis {1}' \
                          ' with size {2}'.format(sub_key, axis, size)
                    raise IndexError(msg)
        else:
            raise TypeError('invalid key {!r}'.format(key))
        return key