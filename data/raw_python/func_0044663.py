def bulk_remove(self, named_graph, add, size=DEFAULT_CHUNK_SIZE):
        """
        Remove batches of statements in n-sized chunks.
        """
        return self.bulk_update(named_graph, add, size, is_add=False)