def reset(self):
        """ Resets (deletes) all indexes. """
        self.dataset_index.reset()
        self.partition_index.reset()
        self.identifier_index.reset()