def decode(self, data, content_type, content_encoding):
        """Deserialize a data stream as serialized using ``encode``
        based on :param:`content_type`.

        :param data: The message data to deserialize.

        :param content_type: The content-type of the data.
            (e.g., ``application/json``).

        :param content_encoding: The content-encoding of the data.
            (e.g., ``utf-8``, ``binary``, or ``us-ascii``).

        :returns: The unserialized data.
        """
        content_type = content_type or 'application/data'
        content_encoding = (content_encoding or 'utf-8').lower()

        # Don't decode 8-bit strings or unicode objects
        if content_encoding not in ('binary', 'ascii-8bit') and \
                not isinstance(data, unicode):
            data = codecs.decode(data, content_encoding)

        try:
            decoder = self._decoders[content_type]
        except KeyError:
            return data

        return decoder(data)