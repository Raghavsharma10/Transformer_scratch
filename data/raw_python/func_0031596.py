def value_to_bytes(self, obj, value, default_endianness=DEFAULT_ENDIANNESS):
        """
        Converts the given value to an appropriately encoded string of bytes that represents it.

        :param obj: The parent :class:`.PebblePacket` of this field
        :type obj: .PebblePacket
        :param value: The python value to serialise.
        :param default_endianness: The default endianness of the value. Used if ``endianness`` was not passed to the
                                   :class:`Field` constructor.
        :type default_endianness: str
        :return: The serialised value
        :rtype: bytes
        """
        return struct.pack(str(self.endianness or default_endianness) + self.struct_format, value)