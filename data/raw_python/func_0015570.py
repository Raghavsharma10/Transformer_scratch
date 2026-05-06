def _decompress(self, compressed_payload):
        '''Decompress a compressed payload into this payload wrapper.
        Note that the decompressed buffer is saved in self._data and the
        counts array is not yet allocated.

        Args:
            compressed_payload (string) a payload in zlib compressed form
        Exception:
            HdrCookieException:
                the compressed payload has an invalid cookie
            HdrLengthException:
                the decompressed size is too small for the HdrPayload structure
                or is not aligned or is too large for the passed payload class
            HdrHistogramSettingsException:
                mismatch in the significant figures, lowest and highest
                         trackable value
        '''
        # make sure this instance is pristine
        if self._data:
            raise RuntimeError('Cannot decompress to an instance with payload')
        # Here it is important to keep a reference to the decompressed
        # string so that it does not get garbage collected
        self._data = zlib.decompress(compressed_payload)
        len_data = len(self._data)

        counts_size = len_data - payload_header_size
        if payload_header_size > counts_size > MAX_COUNTS_SIZE:
            raise HdrLengthException('Invalid size:' + str(len_data))

        # copy the first bytes for the header
        self.payload = PayloadHeader.from_buffer_copy(self._data)

        cookie = self.payload.cookie
        if get_cookie_base(cookie) != V2_ENCODING_COOKIE_BASE:
            raise HdrCookieException('Invalid cookie: %x' % cookie)
        word_size = get_word_size_in_bytes_from_cookie(cookie)
        if word_size != V2_MAX_WORD_SIZE_IN_BYTES:
            raise HdrCookieException('Invalid V2 cookie: %x' % cookie)