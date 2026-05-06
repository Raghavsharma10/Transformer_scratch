def get_multi(self, keys):
        """
        Get multiple keys from server.

        Since keys are converted to b'' when six.PY3 the keys need to be decoded back
        into string . e.g key='test' is read as b'test' and then decoded back to 'test'
        This encode/decode does not work when key is already a six.binary_type hence
        this function remembers which keys were originally sent as str so that
        it only decoded those keys back to string which were sent as string

        :param keys: A list of keys to from server.
        :type keys: list
        :return: A dict with all requested keys.
        :rtype: dict
        """
        # pipeline N-1 getkq requests, followed by a regular getk to uncork the
        # server
        o_keys = keys
        keys, last = keys[:-1], str_to_bytes(keys[-1])
        if six.PY2:
            msg = ''
        else:
            msg = b''
        msg = msg.join([
            struct.pack(self.HEADER_STRUCT +
                        self.COMMANDS['getkq']['struct'] % (len(key)),
                        self.MAGIC['request'],
                        self.COMMANDS['getkq']['command'],
                        len(key), 0, 0, 0, len(key), 0, 0, str_to_bytes(key))
            for key in keys])
        msg += struct.pack(self.HEADER_STRUCT +
                           self.COMMANDS['getk']['struct'] % (len(last)),
                           self.MAGIC['request'],
                           self.COMMANDS['getk']['command'],
                           len(last), 0, 0, 0, len(last), 0, 0, last)

        self._send(msg)

        d = {}
        opcode = -1
        while opcode != self.COMMANDS['getk']['command']:
            (magic, opcode, keylen, extlen, datatype, status, bodylen, opaque,
             cas, extra_content) = self._get_response()

            if status == self.STATUS['success']:
                flags, key, value = struct.unpack('!L%ds%ds' %
                                                  (keylen, bodylen - keylen - 4),
                                                  extra_content)
                if six.PY2:
                    d[key] = self.deserialize(value, flags), cas
                else:
                    try:
                        decoded_key = key.decode()
                    except UnicodeDecodeError:
                        d[key] = self.deserialize(value, flags), cas
                    else:
                        if decoded_key in o_keys:
                            d[decoded_key] = self.deserialize(value, flags), cas
                        else:
                            d[key] = self.deserialize(value, flags), cas

            elif status == self.STATUS['server_disconnected']:
                break
            elif status != self.STATUS['key_not_found']:
                raise MemcachedException('Code: %d Message: %s' % (status, extra_content), status)

        return d