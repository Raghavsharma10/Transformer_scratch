def _pfp__build(self, stream=None, save_offset=False):
        """Build the field and write the result into the stream

        :stream: An IO stream that can be written to
        :returns: None

        """
        if save_offset and stream is not None:
            self._pfp__offset = stream.tell()

        # returns either num bytes written or total data
        res = utils.binary("") if stream is None else 0

        # iterate IN ORDER
        for child in self._pfp__children:
            child_res = child._pfp__build(stream, save_offset)
            res += child_res

        return res