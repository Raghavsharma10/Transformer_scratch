def _unpack_string(self, data):
        """
        Unpacks the next string from the given data

        :param data: A datagram, starting at a string size
        :return: A (string, unread_data) tuple
        """
        # Get the size of the string
        result, data = self._unpack("<H", data)
        size = result[0]

        # Read it
        string_bytes = data[:size]

        # Convert it
        return to_unicode(string_bytes), data[size:]