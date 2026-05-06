def process_chunks(self, chunks):
        """
        Store the incoming chunk at the corresponding position in the
        result array.

        """
        chunk, = chunks
        if chunk.keys:
            self.result[chunk.keys] = chunk.data
        else:
            self.result[...] = chunk.data