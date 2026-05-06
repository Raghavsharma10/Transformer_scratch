def decode(encoded_histogram, b64_wrap=True):
        '''Decode an encoded histogram and return a new histogram instance that
        has been initialized with the decoded content
        Return:
            a new histogram instance representing the decoded content
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
        hdr_payload = HdrHistogramEncoder.decode(encoded_histogram, b64_wrap)
        payload = hdr_payload.payload
        histogram = HdrHistogram(payload.lowest_trackable_value,
                                 payload.highest_trackable_value,
                                 payload.significant_figures,
                                 hdr_payload=hdr_payload)
        return histogram