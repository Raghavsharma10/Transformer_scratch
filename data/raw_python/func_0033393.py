def _get_matchable_segments(segments):
    """
    Performs a depth-first search of the segment tree to get all matchable
    segments.
    """
    for subsegment in segments:
        if isinstance(subsegment, Token):
            break # No tokens allowed next to segments
        if isinstance(subsegment, Segment):
            if isinstance(subsegment, MatchableSegment):
                yield subsegment

            for matchable_subsegment in _get_matchable_segments(subsegment):
                yield matchable_subsegment