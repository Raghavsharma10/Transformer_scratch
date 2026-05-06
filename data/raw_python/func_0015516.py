def delete(self, key, cas=0):
        """
        Delete a key/value from server. If key existed and was deleted, return True.

        :param key: Key's name to be deleted
        :type key: six.string_types
        :param cas: If set, only delete the key if its CAS value matches.
        :type cas: int
        :return: True in case o success and False in case of failure.
        :rtype: bool
        """
        logger.debug('Deleting key %s', key)
        self._send(struct.pack(self.HEADER_STRUCT +
                               self.COMMANDS['delete']['struct'] % len(key),
                               self.MAGIC['request'],
                               self.COMMANDS['delete']['command'],
                               len(key), 0, 0, 0, len(key), 0, cas, str_to_bytes(key)))

        (magic, opcode, keylen, extlen, datatype, status, bodylen, opaque,
         cas, extra_content) = self._get_response()

        if status == self.STATUS['server_disconnected']:
            return False
        if status != self.STATUS['success'] and status not in (self.STATUS['key_not_found'], self.STATUS['key_exists']):
            raise MemcachedException('Code: %d message: %s' % (status, extra_content), status)

        logger.debug('Key deleted %s', key)
        return status != self.STATUS['key_exists']