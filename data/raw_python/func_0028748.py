def output(self, chunk):
        """
        Dispatch the given Chunk onto all the registered output queues.

        If the chunk is None, it is silently ignored.

        """
        if chunk is not None:
            for queue in self.output_queues:
                queue.put(chunk)