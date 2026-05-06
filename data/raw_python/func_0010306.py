def get_data(self):
        """Get the data associated with this filedata object

        :returns: Data associated with this object or None if none exists
        :rtype: str (Python2)/bytes (Python3) or None

        """
        # NOTE: we assume that the "embed" option is used
        base64_data = self._json_data.get("fdData")
        if base64_data is None:
            return None
        else:
            # need to convert to bytes() with python 3
            return base64.decodestring(six.b(base64_data))