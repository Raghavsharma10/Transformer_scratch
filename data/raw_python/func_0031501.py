def decompress(self, chunk):
        """Decompress the chunk of data.

        :param bytes chunk: data chunk

        :rtype: bytes
        """

        try:
            return self._decompressobj.decompress(chunk)
        except zlib.error:
            # ugly hack to work with raw deflate content that may
            # be sent by microsoft servers. For more information, see:
            # http://carsten.codimi.de/gzip.yaws/
            # http://www.port80software.com/200ok/archive/2005/10/31/868.aspx
            # http://www.gzip.org/zlib/zlib_faq.html#faq38
            if self._first_chunk:
                self._decompressobj = zlib.decompressobj(-zlib.MAX_WBITS)
                return self._decompressobj.decompress(chunk)

            raise
        finally:
            self._first_chunk = False