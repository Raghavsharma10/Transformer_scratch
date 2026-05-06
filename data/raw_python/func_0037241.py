def sendfile(self, data, zlib_compress=None, compress_level=6):
        """Send data from a file object"""
        if hasattr(data, 'seek'):
            data.seek(0)

        chunk_size = CHUNK_SIZE

        if zlib_compress:
            chunk_size = BLOCK_SIZE
            compressor = compressobj(compress_level)

        while 1:
            binarydata = data.read(chunk_size)
            if binarydata == '':
                break
            if zlib_compress:
                binarydata = compressor.compress(binarydata)
                if not binarydata:
                    continue
            self.send(binarydata)

        if zlib_compress:
            remaining = compressor.flush()
            while remaining:
                binarydata = remaining[:BLOCK_SIZE]
                remaining = remaining[BLOCK_SIZE:]
                self.send(binarydata)