def junction_overlap(self,tx,tolerance=0):
    """Calculate the junction overlap between two transcripts

    :param tx: Other transcript
    :type tx: Transcript
    :param tolerance: how close to consider two junctions as overlapped (default=0)
    :type tolerance: int
    :return: Junction Overlap Report
    :rtype: Transcript.JunctionOverlap
    """
    self._initialize()
    return JunctionOverlap(self,tx,tolerance)