def all(self):
        """ Returns list with vids of all indexed partitions. """
        partitions = []

        query = text("""
            SELECT dataset_vid, vid
            FROM partition_index;""")

        for result in self.execute(query):
            dataset_vid, vid = result
            partitions.append(PartitionSearchResult(dataset_vid=dataset_vid, vid=vid, score=1))
        return partitions