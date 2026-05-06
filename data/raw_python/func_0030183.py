def index_partition(self, partition, force=False):
        """ Adds given partition to the index. """
        self.backend.partition_index.index_one(partition, force=force)