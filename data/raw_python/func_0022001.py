def _wrap_in_jws(self, nonce, obj):
        """
        Wrap ``JSONDeSerializable`` object in JWS.

        ..  todo:: Implement ``acmePath``.

        :param ~josepy.interfaces.JSONDeSerializable obj:
        :param bytes nonce:

        :rtype: `bytes`
        :return: JSON-encoded data
        """
        with LOG_JWS_SIGN(key_type=self._key.typ, alg=self._alg.name,
                          nonce=nonce):
            jobj = obj.json_dumps().encode()
            return (
                JWS.sign(
                    payload=jobj, key=self._key, alg=self._alg, nonce=nonce)
                .json_dumps()
                .encode())