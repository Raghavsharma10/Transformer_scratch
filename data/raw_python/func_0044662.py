def bulk_add(self, named_graph, add, size=DEFAULT_CHUNK_SIZE):
        """
        Add batches of statements in n-sized chunks.
        """
        return self.bulk_update(named_graph, add, size)