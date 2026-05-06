def delete_multi(self, keys):
        """
        Delete multiple keys from server in one command.

        :param keys: A list of keys to be deleted
        :type keys: list
        :return: True in case of success and False in case of failure.
        :rtype: bool
        """
        logger.debug('Deleting keys %r', keys)
        if six.PY2:
            msg = ''
        else:
            msg = b''
        for key in keys:
            msg += struct.pack(
                self.HEADER_STRUCT +
                self.COMMANDS['delete']['struct'] % len(key),
                self.MAGIC['request'],
                self.COMMANDS['delete']['command'],
                len(key), 0, 0, 0, len(key), 0, 0, str_to_bytes(key))

        msg += struct.pack(
            self.HEADER_STRUCT +
            self.COMMANDS['noop']['struct'],
            self.MAGIC['request'],
            self.COMMANDS['noop']['command'],
            0, 0, 0, 0, 0, 0, 0)

        self._send(msg)

        opcode = -1
        retval = True
        while opcode != self.COMMANDS['noop']['command']:
            (magic, opcode, keylen, extlen, datatype, status, bodylen, opaque,
             cas, extra_content) = self._get_response()
            if status != self.STATUS['success']:
                retval = False
            if status == self.STATUS['server_disconnected']:
                break

        return retval