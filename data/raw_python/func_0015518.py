def flush_all(self, time):
        """
        Send a command to server flush|delete all keys.

        :param time: Time to wait until flush in seconds.
        :type time: int
        :return: True in case of success, False in case of failure
        :rtype: bool
        """
        logger.info('Flushing memcached')
        self._send(struct.pack(self.HEADER_STRUCT +
                               self.COMMANDS['flush']['struct'],
                               self.MAGIC['request'],
                               self.COMMANDS['flush']['command'],
                               0, 4, 0, 0, 4, 0, 0, time))

        (magic, opcode, keylen, extlen, datatype, status, bodylen, opaque,
         cas, extra_content) = self._get_response()

        if status not in (self.STATUS['success'], self.STATUS['server_disconnected']):
            raise MemcachedException('Code: %d message: %s' % (status, extra_content), status)

        logger.debug('Memcached flushed')
        return True