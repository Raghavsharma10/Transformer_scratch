def SegmentCollection(mode="agg-fast", *args, **kwargs):
    """
    mode: string
      - "raw" (speed: fastest, size: small, output: ugly, no dash,
               no thickness)
      - "agg" (speed: slower, size: medium, output: perfect, no dash)
    """

    if mode == "raw":
        return RawSegmentCollection(*args, **kwargs)
    return AggSegmentCollection(*args, **kwargs)