def read(self, buf=None, n=None):
        """Reads a specified number of ``bytes`` from this stream.

        arg:    n (cardinal): the number of ``bytes`` to read
        return: (integer) - the ``bytes`` read
        raise:  IllegalState - this stream has been closed or
                ``at_end_of_stream()`` is ``true``
        raise:  InvalidArgument - the size of ``buf`` is less than ``n``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        if n is not None:
            return self._my_data.read(n)
        else:
            return self._my_data.read()