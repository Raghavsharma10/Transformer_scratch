def set_multi(self, mappings, time=100, compress_level=-1):
        """
        Set multiple keys with its values on server.

        If a key is a (key, cas) tuple, insert as if cas(key, value, cas) had
        been called.

        :param mappings: A dict with keys/values
        :type mappings: dict
        :param time: Time in seconds that your key will expire.
        :type time: int
        :param compress_level: How much to compress.
            0 = no compression, 1 = fastest, 9 = slowest but best,
            -1 = default compression level.
        :type compress_level: int
        :return: True
        :rtype: bool
        """
        mappings = mappings.items()
        msg = []

        for key, value in mappings:
            if isinstance(key, tuple):
                key, cas = key
            else:
                cas = None

            if cas == 0:
                # Like cas(), if the cas value is 0, treat it as compare-and-set against not
                # existing.
                command = 'addq'
            else:
                command = 'setq'

            flags, value = self.serialize(value, compress_level=compress_level)
            m = struct.pack(self.HEADER_STRUCT +
                            self.COMMANDS[command]['struct'] % (len(key), len(value)),
                            self.MAGIC['request'],
                            self.COMMANDS[command]['command'],
                            len(key),
                            8, 0, 0, len(key) + len(value) + 8, 0, cas or 0,
                            flags, time, str_to_bytes(key), value)
            msg.append(m)

        m = struct.pack(self.HEADER_STRUCT +
                        self.COMMANDS['noop']['struct'],
                        self.MAGIC['request'],
                        self.COMMANDS['noop']['command'],
                        0, 0, 0, 0, 0, 0, 0)
        msg.append(m)

        if six.PY2:
            msg = ''.join(msg)
        else:
            msg = b''.join(msg)

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