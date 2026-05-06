def _pfp__build(self, stream=None, save_offset=False):
        """Build the String field

        :stream: TODO
        :returns: TODO

        """
        if stream is not None and save_offset:
            self._pfp__offset = stream.tell()

        data = self._pfp__value + utils.binary("\x00")
        if stream is None:
            return data
        else:
            stream.write(data)
            return len(data)