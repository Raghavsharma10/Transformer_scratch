def read_to_buffer(self, buf, n):
        """Reads a specified number of ``bytes`` from this stream.

        arg:    buf (byte[]): the buffer in which the data is read
        arg:    n (cardinal): the number of ``bytes`` to read
        return: (integer) - the actual number of ``bytes`` read
        raise:  IllegalState - this stream has been closed or
                ``at_end_of_stream()`` is ``true``
        raise:  InvalidArgument - the size of ``buf`` is less than ``n``
        raise:  NullArgument - ``buf`` is ``null``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        if buf is None:
            raise NullArgument()
        if self._my_data.closed or self.at_end_of_stream():
            raise IllegalState()
        initial_buf_len = len(buf)
        buf.append(self._my_data.read(size=n))
        return len(buf) - initial_buf_len