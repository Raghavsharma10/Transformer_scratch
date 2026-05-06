def _pfp__parse(self, stream, save_offset=False):
        """Parse the incoming stream

        :stream: Input stream to be parsed
        :returns: Number of bytes parsed
        """
        if save_offset:
            self._pfp__offset = stream.tell()

        max_res = 0
        for child in self._pfp__children:
            child_res = child._pfp__parse(stream, save_offset)
            if child_res > max_res:
                max_res = child_res

            # rewind the stream
            stream.seek(child_res, -1)
        self._pfp__size = max_res

        self._pfp__buff = six.BytesIO(stream.read(self._pfp__size))
        return max_res