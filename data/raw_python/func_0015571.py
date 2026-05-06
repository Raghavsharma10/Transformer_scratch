def compress(self, counts_limit):
        '''Compress this payload instance
        Args:
            counts_limit how many counters should be encoded
                           starting from index 0 (can be 0),
        Return:
            the compressed payload (python string)
        '''
        if self.payload:
            # worst case varint encoded length is when each counter is at the maximum value
            # in this case 1 more byte per counter is needed due to the more bits
            varint_len = counts_limit * (self.word_size + 1)
            # allocate enough space to fit the header and the varint string
            encode_buf = (c_byte * (payload_header_size + varint_len))()

            # encode past the payload header
            varint_len = encode(addressof(self.counts), counts_limit,
                                self.word_size,
                                addressof(encode_buf) + payload_header_size,
                                varint_len)

            # copy the header after updating the varint stream length
            self.payload.payload_len = varint_len
            ctypes.memmove(addressof(encode_buf), addressof(self.payload), payload_header_size)

            cdata = zlib.compress(ctypes.string_at(encode_buf, payload_header_size + varint_len))
            return cdata
        # can't compress if no payload
        raise RuntimeError('No payload to compress')