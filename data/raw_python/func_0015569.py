def init_counts(self, counts_len):
        '''Called after instantiating with a compressed payload
        Params:
            counts_len counts size to use based on decoded settings in the header
        '''
        assert self._data and counts_len and self.counts_len == 0
        self.counts_len = counts_len
        self._init_counts()

        results = decode(self._data, payload_header_size, addressof(self.counts),
                         counts_len, self.word_size)
        # no longer needed
        self._data = None
        return results