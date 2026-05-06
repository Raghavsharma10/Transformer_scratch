def stats(self, key=None):
        """
        Return server stats.

        :param key: Optional if you want status from a key.
        :type key: six.string_types
        :return: A dict with server stats
        :rtype: dict
        """
        # TODO: Stats with key is not working.
        if key is not None:
            if isinstance(key, text_type):
                key = str_to_bytes(key)
            keylen = len(key)
            packed = struct.pack(
                self.HEADER_STRUCT + '%ds' % keylen,
                self.MAGIC['request'],
                self.COMMANDS['stat']['command'],
                keylen, 0, 0, 0, keylen, 0, 0, key)
        else:
            packed = struct.pack(
                self.HEADER_STRUCT,
                self.MAGIC['request'],
                self.COMMANDS['stat']['command'],
                0, 0, 0, 0, 0, 0, 0)

        self._send(packed)

        value = {}

        while True:
            response = self._get_response()

            status = response[5]
            if status == self.STATUS['server_disconnected']:
                break

            keylen = response[2]
            bodylen = response[6]

            if keylen == 0 and bodylen == 0:
                break

            extra_content = response[-1]
            key = extra_content[:keylen]
            body = extra_content[keylen:bodylen]
            value[key.decode() if isinstance(key, bytes) else key] = body

        return value