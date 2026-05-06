def _pfp__width(self):
        """Return the width of the field (sizeof)
        """
        raw_output = six.BytesIO()
        output = bitwrap.BitwrappedStream(raw_output)
        self._pfp__build(output)
        output.flush()
        return len(raw_output.getvalue())