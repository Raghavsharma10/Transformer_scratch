def process(texts, *args, **kwargs):
    """
    Processes a single sequence of texts with a
    :class:`~deltas.SegmentMatcher`.

    :Parameters:
        texts : `iterable`(`str`)
            sequence of texts
        args : `tuple`
            passed to :class:`~deltas.SegmentMatcher`'s
            constructor
        kwaths : `dict`
            passed to :class:`~deltas.SegmentMatcher`'s
            constructor
    """
    processor = SegmentMatcher.Processor(*args, **kwargs)
    for text in texts:
        yield processor.process(text)