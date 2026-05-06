def _unpack(self, fmt, data):
        """
        Calls struct.unpack().

        Returns a tuple containing the result tuple and the subset of data
        containing the unread content.

        :param fmt: The format of data
        :param data: Data to unpack
        :return: A tuple (result tuple, unread_data)
        """
        size = struct.calcsize(fmt)
        read, unread = data[:size], data[size:]
        return struct.unpack(fmt, read), unread