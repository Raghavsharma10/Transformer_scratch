def decode(encoded_histogram, b64_wrap=True):
        '''Decode a wire histogram encoding into a read-only Hdr Payload instance
        Args:
            encoded_histogram a string containing the wire encoding of a histogram
                              such as one returned from encode()
        Returns:
            an hdr_payload instance with all the decoded/uncompressed fields

        Exception:
            TypeError in case of base64 decode error
            HdrCookieException:
                the main header has an invalid cookie
                the compressed payload header has an invalid cookie
            HdrLengthException:
                the decompressed size is too small for the HdrPayload structure
                or is not aligned or is too large for the passed payload class
            HdrHistogramSettingsException:
                mismatch in the significant figures, lowest and highest
                         trackable value
            zlib.error:
                in case of zlib decompression error
        '''
        if b64_wrap:
            b64decode = base64.b64decode(encoded_histogram)
            # this string has 2 parts in it: the header (raw) and the payload (compressed)
            b64dec_len = len(b64decode)

            if b64dec_len < ext_header_size:
                raise HdrLengthException('Base64 decoded message too short')

            header = ExternalHeader.from_buffer_copy(b64decode)
            if get_cookie_base(header.cookie) != V2_COMPRESSION_COOKIE_BASE:
                raise HdrCookieException()
            if header.length != b64dec_len - ext_header_size:
                raise HdrLengthException('Decoded length=%d buffer length=%d' %
                                         (header.length, b64dec_len - ext_header_size))
            # this will result in a copy of the compressed payload part
            # could not find a way to do otherwise since zlib.decompress()
            # expects a string (and does not like a buffer or a memoryview object)
            cpayload = b64decode[ext_header_size:]
        else:
            cpayload = encoded_histogram
        hdr_payload = HdrPayload(8, compressed_payload=cpayload)
        return hdr_payload