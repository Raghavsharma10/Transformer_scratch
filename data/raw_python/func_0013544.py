def getStats(self):
        """
        Returns the GA4GH protocol representation of this read group's
        ReadStats.
        """
        stats = protocol.ReadStats()
        stats.aligned_read_count = self.getNumAlignedReads()
        stats.unaligned_read_count = self.getNumUnalignedReads()
        # TODO base_count requires iterating through all reads
        return stats