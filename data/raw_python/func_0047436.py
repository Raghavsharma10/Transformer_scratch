def target_range(self):
    """Get the range covered on the target/reference strand

    :returns: Genomic range of the target strand
    :rtype: GenomicRange

    """
    a = self.alignment_ranges
    return GenomicRange(a[0][0].chr,a[0][0].start,a[-1][0].end)