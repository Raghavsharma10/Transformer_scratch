def all(self):
        """ Returns list with all indexed partitions. """
        partitions = []
        for partition in self.index.searcher().documents():
            partitions.append(
                PartitionSearchResult(dataset_vid=partition['dataset_vid'], vid=partition['vid'], score=1))
        return partitions