def noop(self):
        """
        Send a NOOP command

        :return: Returns the status.
        :rtype: int
        """
        logger.debug('Sending NOOP')
        data = struct.pack(self.HEADER_STRUCT +
                           self.COMMANDS['noop']['struct'],
                           self.MAGIC['request'],
                           self.COMMANDS['noop']['command'],
                           0, 0, 0, 0, 0, 0, 0)
        self._send(data)

        (magic, opcode, keylen, extlen, datatype, status, bodylen, opaque,
         cas, extra_content) = self._get_response()

        logger.debug('Value Length: %d. Body length: %d. Data type: %d',
                     extlen, bodylen, datatype)

        if status != self.STATUS['success']:
            logger.debug('NOOP failed (status is %d). Message: %s' % (status, extra_content))

        return int(status)