def parse(cls, message, default_endianness=DEFAULT_ENDIANNESS):
        """
        Parses a message without any framing, returning the decoded result and length of message consumed. The result
        will always be of the same class as :meth:`parse` was called on. If the message is invalid,
        :exc:`.PacketDecodeError` will be raised.

        :param message: The message to decode.
        :type message: bytes
        :param default_endianness: The default endianness, unless overridden by the fields or class metadata.
                                   Should usually be left at ``None``. Otherwise, use ``'<'`` for little endian and
                                   ``'>'`` for big endian.
        :return: ``(decoded_message, decoded length)``
        :rtype: (:class:`PebblePacket`, :any:`int`)
        """
        obj = cls()
        offset = 0
        if hasattr(cls, '_Meta'):
            default_endianness = cls._Meta.get('endianness', default_endianness)
        for k, v in iteritems(cls._type_mapping):
            try:
                value, length = v.buffer_to_value(obj, message, offset, default_endianness=default_endianness)
            except Exception:
                logger.warning("Exception decoding {}.{}".format(cls.__name__, k))
                raise
            offset += length
            setattr(obj, k, value)
        return obj, offset