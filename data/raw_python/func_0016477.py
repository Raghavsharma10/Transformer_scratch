def encode(self, data, serializer=None):
        """
        Serialize a data structure into a string suitable for sending
        as an AMQP message body.

        :param data: The message data to send. Can be a list,
            dictionary or a string.

        :keyword serializer: An optional string representing
            the serialization method you want the data marshalled
            into. (For example, ``json``, ``raw``, or ``pickle``).

            If ``None`` (default), then `JSON`_ will be used, unless
            ``data`` is a ``str`` or ``unicode`` object. In this
            latter case, no serialization occurs as it would be
            unnecessary.

            Note that if ``serializer`` is specified, then that
            serialization method will be used even if a ``str``
            or ``unicode`` object is passed in.

        :returns: A three-item tuple containing the content type
            (e.g., ``application/json``), content encoding, (e.g.,
            ``utf-8``) and a string containing the serialized
            data.

        :raises SerializerNotInstalled: If the serialization method
              requested is not available.
        """
        if serializer == "raw":
            return raw_encode(data)
        if serializer and not self._encoders.get(serializer):
            raise SerializerNotInstalled(
                        "No encoder installed for %s" % serializer)

        # If a raw string was sent, assume binary encoding
        # (it's likely either ASCII or a raw binary file, but 'binary'
        # charset will encompass both, even if not ideal.
        if not serializer and isinstance(data, str):
            # In Python 3+, this would be "bytes"; allow binary data to be
            # sent as a message without getting encoder errors
            return "application/data", "binary", data

        # For unicode objects, force it into a string
        if not serializer and isinstance(data, unicode):
            payload = data.encode("utf-8")
            return "text/plain", "utf-8", payload

        if serializer:
            content_type, content_encoding, encoder = \
                    self._encoders[serializer]
        else:
            encoder = self._default_encode
            content_type = self._default_content_type
            content_encoding = self._default_content_encoding

        payload = encoder(data)
        return content_type, content_encoding, payload