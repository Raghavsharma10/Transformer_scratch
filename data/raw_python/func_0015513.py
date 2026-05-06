def _incr_decr(self, command, key, value, default, time):
        """
        Function which increments and decrements.

        :param key: Key's name
        :type key: six.string_types
        :param value: Number to be (de|in)cremented
        :type value: int
        :param default: Default value if key does not exist.
        :type default: int
        :param time: Time in seconds to expire key.
        :type time: int
        :return: Actual value of the key on server
        :rtype: int
        """
        time = time if time >= 0 else self.MAXIMUM_EXPIRE_TIME
        self._send(struct.pack(self.HEADER_STRUCT +
                               self.COMMANDS[command]['struct'] % len(key),
                               self.MAGIC['request'],
                               self.COMMANDS[command]['command'],
                               len(key),
                               20, 0, 0, len(key) + 20, 0, 0, value,
                               default, time, str_to_bytes(key)))

        (magic, opcode, keylen, extlen, datatype, status, bodylen, opaque,
         cas, extra_content) = self._get_response()

        if status not in (self.STATUS['success'], self.STATUS['server_disconnected']):
            raise MemcachedException('Code: %d Message: %s' % (status, extra_content), status)
        if status == self.STATUS['server_disconnected']:
            return 0

        return struct.unpack('!Q', extra_content)[0]