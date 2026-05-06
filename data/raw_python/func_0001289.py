def raw(self):
        """Return raw key.

        returns:
            str: raw key
        """
        if self._raw:
            return text_type(self._raw).strip("\r\n")
        else:
            return text_type(base64decode(self._b64encoded)).strip("\r\n")