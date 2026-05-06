def parse(requirements):
    """
    Parses given requirements line-by-line.
    """
    transformer = RTransformer()
    return map(transformer.transform, filter(None, map(_parse, requirements.splitlines())))