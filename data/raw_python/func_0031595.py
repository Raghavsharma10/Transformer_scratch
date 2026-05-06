def buffer_to_value(self, obj, buffer, offset, default_endianness=DEFAULT_ENDIANNESS):
        """
        Converts the bytes in ``buffer`` at ``offset`` to a native Python value. Returns that value and the number of
        bytes consumed to create it.

        :param obj: The parent :class:`.PebblePacket` of this field
        :type obj: .PebblePacket
        :param buffer: The buffer from which to extract a value.
        :type buffer: bytes
        :param offset: The offset in the buffer to start at.
        :type offset: int
        :param default_endianness: The default endianness of the value. Used if ``endianness`` was not passed to the
                                   :class:`Field` constructor.
        :type default_endianness: str
        :return: (value, length)
        :rtype: (:class:`object`, :any:`int`)
        """
        try:
            value, length = struct.unpack_from(str(self.endianness or default_endianness)
                                      + self.struct_format, buffer, offset)[0], struct.calcsize(self.struct_format)
            if self._enum is not None:
                try:
                    return self._enum(value), length
                except ValueError as e:
                    raise PacketDecodeError("{}: {}".format(self.type, e))
            else:
                return value, length
        except struct.error as e:
            raise PacketDecodeError("{}: {}".format(self.type, e))