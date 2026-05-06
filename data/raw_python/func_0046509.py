def range(self):
    """Return the range the transcript loci covers

    :return: range
    :rtype: GenomicRange
    """
    chrs = set([x.range.chr for x in self.get_transcripts()])
    if len(chrs) != 1: return None
    start = min([x.range.start for x in self.get_transcripts()])
    end = max([x.range.end for x in self.get_transcripts()])
    return GenomicRange(list(chrs)[0],start,end)