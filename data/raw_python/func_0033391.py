def diff_segments(a_segments, b_segments):
    """
    Performs a diff comparison between two pre-clustered
    :class:`deltas.Segment` trees.  In most cases, segmentation
    takes 100X more time than actually performing the diff.

    :Parameters:
        a_segments : :class:`deltas.Segment`
            An initial sequence
        b_segments : :class:`deltas.Segment`
            A changed sequence

    :Returns:
        An `iterable` of operations.
    """
    # Match and re-sequence unmatched tokens
    a_segment_tokens, b_segment_tokens = _cluster_matching_segments(a_segments,
                                                                    b_segments)

    # Perform a simple LCS over unmatched tokens and clusters
    clustered_ops = sequence_matcher.diff(a_segment_tokens, b_segment_tokens)

    # Return the expanded (de-clustered) operations
    return (op for op in SegmentOperationsExpander(clustered_ops,
                                                   a_segment_tokens,
                                                   b_segment_tokens).expand())