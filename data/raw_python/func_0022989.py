def PointCollection(mode="raw", *args, **kwargs):
    """
    mode: string
      - "raw"  (speed: fastest, size: small,   output: ugly)
      - "agg"  (speed: fast,    size: small,   output: beautiful)
    """

    if mode == "raw":
        return RawPointCollection(*args, **kwargs)
    return AggPointCollection(*args, **kwargs)