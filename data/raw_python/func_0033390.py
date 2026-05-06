def diff(a, b, segmenter=None):
    """
    Performs a diff comparison between two sequences of tokens (`a` and `b`)
    using `segmenter` to cluster and match
    :class:`deltas.MatchableSegment`.

    :Example:
        >>> from deltas import segment_matcher, text_split
        >>>
        >>> a = text_split.tokenize("This is some text.  This is some other text.")
        >>> b = text_split.tokenize("This is some other text.  This is some text.")
        >>> operations = segment_matcher.diff(a, b)
        >>>
        >>> for op in operations:
        ...     print(op.name, repr(''.join(a[op.a1:op.a2])),
        ...           repr(''.join(b[op.b1:op.b2])))
        ...
        equal 'This is some other text.' 'This is some other text.'
        insert '' '  '
        equal 'This is some text.' 'This is some text.'
        delete '  ' ''

    :Parameters:
        a : `list`(:class:`deltas.tokenizers.Token`)
            Initial sequence
        b : `list`(:class:`deltas.tokenizers.Token`)
            Changed sequence
        segmenter : :class:`deltas.Segmenter`
            A segmenter to use on the tokens.

    :Returns:
        An `iterable` of operations.
    """
    a, b = list(a), list(b)
    segmenter = segmenter or SEGMENTER

    # Cluster the input tokens
    a_segments = segmenter.segment(a)
    b_segments = segmenter.segment(b)

    return diff_segments(a_segments, b_segments)