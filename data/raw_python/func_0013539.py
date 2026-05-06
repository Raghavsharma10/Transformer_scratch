def getStats(self):
        """
        Returns the GA4GH protocol representation of this read group set's
        ReadStats.
        """
        stats = protocol.ReadStats()
        stats.aligned_read_count = self._numAlignedReads
        stats.unaligned_read_count = self._numUnalignedReads
        return stats