def get(self, key):
        """
        Get a key and its CAS value from server.  If the value isn't cached, return
        (None, None).

        :param key: Key's name
        :type key: six.string_types
        :return: Returns (value, cas).
        :rtype: object
        """
        logger.debug('Getting key %s', key)
        data = struct.pack(self.HEADER_STRUCT +
                           self.COMMANDS['get']['struct'] % (len(key)),
                           self.MAGIC['request'],
                           self.COMMANDS['get']['command'],
                           len(key), 0, 0, 0, len(key), 0, 0, str_to_bytes(key))
        self._send(data)

        (magic, opcode, keylen, extlen, datatype, status, bodylen, opaque,
         cas, extra_content) = self._get_response()

        logger.debug('Value Length: %d. Body length: %d. Data type: %d',
                     extlen, bodylen, datatype)

        if status != self.STATUS['success']:
            if status == self.STATUS['key_not_found']:
                logger.debug('Key not found. Message: %s', extra_content)
                return None, None

            if status == self.STATUS['server_disconnected']:
                return None, None

            raise MemcachedException('Code: %d Message: %s' % (status, extra_content), status)

        flags, value = struct.unpack('!L%ds' % (bodylen - 4, ), extra_content)

        return self.deserialize(value, flags), cas