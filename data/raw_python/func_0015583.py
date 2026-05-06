def decode_and_add(self, encoded_histogram):
        '''Decode an encoded histogram and add it to this histogram
        Args:
            encoded_histogram (string) an encoded histogram
                following the V1 format, such as one returned by the encode() method
        Exception:
            TypeError in case of base64 decode error
            HdrCookieException:
                the main header has an invalid cookie
                the compressed payload header has an invalid cookie
            HdrLengthException:
                the decompressed size is too small for the HdrPayload structure
                or is not aligned or is too large for the passed payload class
            zlib.error:
                in case of zlib decompression error
        '''
        other_hist = HdrHistogram.decode(encoded_histogram, self.b64_wrap)
        self.add(other_hist)