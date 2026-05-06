def _decompress(self, fp):
        """
        Internal function for decompressing a backup file with the DEFLATE algorithm

        :rtype: Proxy
        """
        decompressor = zlib.decompressobj()
        if self.stream:
            return Proxy(decompressor.decompress, fp)
        else:
            out = io.BytesIO(decompressor.decompress(fp.read()))
            out.write(decompressor.flush())
            out.seek(0)
            return out