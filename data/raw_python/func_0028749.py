def run(self):
        """
        Emit the Chunk instances which cover the underlying Array.

        The Array is divided into chunks with a size limit of
        MAX_CHUNK_SIZE which are emitted into all registered output
        queues.

        """
        try:
            chunk_index = self.chunk_index_gen(self.array.shape,
                                               self.iteration_order)
            for key in chunk_index:
                # Now we have the slices that describe the next chunk.
                # For example, key might be equivalent to
                # `[11:12, 0:3, :, :]`.
                # Simply "realise" the data for that region and emit it
                # as a Chunk to all registered output queues.
                if self.masked:
                    data = self.array[key].masked_array()
                else:
                    data = self.array[key].ndarray()
                output_chunk = Chunk(key, data)
                self.output(output_chunk)
        except:
            self.abort()
            raise
        else:
            for queue in self.output_queues:
                queue.put(QUEUE_FINISHED)