def encode(self):
        '''Compress the associated encodable payload,
        prepend the header then encode with base64 if requested

        Returns:
            the b64 encoded wire encoding of the histogram (as a string)
            or the compressed payload (as a string, if b64 wrappinb is disabled)
        '''
        # only compress the first non zero buckets
        # if histogram is empty we do not encode any counter
        if self.histogram.total_count:
            relevant_length = \
                self.histogram.get_counts_array_index(self.histogram.max_value) + 1
        else:
            relevant_length = 0
        cpayload = self.payload.compress(relevant_length)
        if self.b64_wrap:
            self.header.length = len(cpayload)
            header_str = ctypes.string_at(addressof(self.header), ext_header_size)
            return base64.b64encode(header_str + cpayload)
        return cpayload