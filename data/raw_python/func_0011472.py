def set_bitfield_padded(self, val):
        """Set if the bitfield input/output stream should be padded

        :val: True/False
        :returns: None
        """
        self._padded_bitfield = val
        self._stream.padded = val
        self._ctxt._pfp__padded_bitfield = val