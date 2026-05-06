def _set_add_replace(self, command, key, value, time, cas=0, compress_level=-1):
        """
        Function to set/add/replace commands.

        :param key: Key's name
        :type key: six.string_types
        :param value: A value to be stored on server.
        :type value: object
        :param time: Time in seconds that your key will expire.
        :type time: int
        :param cas: The CAS value that must be matched for this operation to complete, or 0 for no CAS.
        :type cas: int
        :param compress_level: How much to compress.
            0 = no compression, 1 = fastest, 9 = slowest but best,
            -1 = default compression level.
        :type compress_level: int
        :return: True in case of success and False in case of failure
        :rtype: bool
        """
        time = time if time >= 0 else self.MAXIMUM_EXPIRE_TIME
        logger.debug('Setting/adding/replacing key %s.', key)
        flags, value = self.serialize(value, compress_level=compress_level)
        logger.debug('Value bytes %s.', len(value))
        if isinstance(value, text_type):
            value = value.encode('utf8')

        self._send(struct.pack(self.HEADER_STRUCT +
                               self.COMMANDS[command]['struct'] % (len(key), len(value)),
                               self.MAGIC['request'],
                               self.COMMANDS[command]['command'],
                               len(key), 8, 0, 0, len(key) + len(value) + 8, 0, cas, flags,
                               time, str_to_bytes(key), value))

        (magic, opcode, keylen, extlen, datatype, status, bodylen, opaque,
         cas, extra_content) = self._get_response()

        if status != self.STATUS['success']:
            if status == self.STATUS['key_exists']:
                return False
            elif status == self.STATUS['key_not_found']:
                return False
            elif status == self.STATUS['server_disconnected']:
                return False
            raise MemcachedException('Code: %d Message: %s' % (status, extra_content), status)

        return True