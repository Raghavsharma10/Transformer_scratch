def PathCollection(mode="agg", *args, **kwargs):
    """
    mode: string
      - "raw"   (speed: fastest, size: small, output: ugly, no dash,
                 no thickness)
      - "agg"   (speed: medium, size: medium output: nice, some flaws, no dash)
      - "agg+"  (speed: slow, size: big, output: perfect, no dash)
    """

    if mode == "raw":
        return RawPathCollection(*args, **kwargs)
    elif mode == "agg+":
        return AggPathCollection(*args, **kwargs)
    return AggFastPathCollection(*args, **kwargs)