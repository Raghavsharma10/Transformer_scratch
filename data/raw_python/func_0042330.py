def iter_chunks(self, start_count=0):
        """
        Iterate over the chunks of the file according to their length prefixes.
        yields: index <int>, encrypted chunks without length prefixes <bytes>, lastchunk <bool>
        """
        ciphertext = self.chunks_block
        chunknum = start_count
        idx = 0
        lastchunk = False
        while idx < len(ciphertext):
            plainlen = int.from_bytes(ciphertext[idx: idx+4], 'little')
            chunklen = plainlen + 16
            if idx + 4 + chunklen == len(ciphertext):
                lastchunk = True
            elif idx + chunklen > len(ciphertext):
                raise ValueError("Bad ciphertext; when reading chunks, hit EOF early")
            yield chunknum, ciphertext[idx + 4 : idx + 4 + chunklen], lastchunk
            idx += chunklen + 4
            chunknum += 1