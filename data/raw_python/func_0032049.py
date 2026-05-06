def serialise(self, default_endianness=None):
        """
        Serialise a message, without including any framing.

        :param default_endianness: The default endianness, unless overridden by the fields or class metadata.
                                   Should usually be left at ``None``. Otherwise, use ``'<'`` for little endian and
                                   ``'>'`` for big endian.
        :type default_endianness: str
        :return: The serialised message.
        :rtype: bytes
        """
        # Figure out an endianness.
        endianness = (default_endianness or DEFAULT_ENDIANNESS)
        if hasattr(self, '_Meta'):
            endianness = self._Meta.get('endianness', endianness)

        inferred_fields = set()
        for k, v in iteritems(self._type_mapping):
            inferred_fields |= {x._name for x in v.dependent_fields()}
        for field in inferred_fields:
            setattr(self, field, None)

        # Some fields want to manipulate other fields that appear before them (e.g. Unions)
        for k, v in iteritems(self._type_mapping):
            v.prepare(self, getattr(self, k))

        message = b''
        for k, v in iteritems(self._type_mapping):
            message += v.value_to_bytes(self, getattr(self, k), default_endianness=endianness)
        return message