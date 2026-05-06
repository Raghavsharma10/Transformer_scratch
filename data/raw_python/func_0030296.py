def _build_join(t):
    """ Populates join token fields. """
    t.source.name = t.source.parsed_name
    t.source.alias = t.source.parsed_alias[0] if t.source.parsed_alias else ''

    return t