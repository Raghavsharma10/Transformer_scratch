def _flat_alias(t):
    """ Populates token (column or table) fields from parse result. """
    t.name = t.parsed_name
    t.alias = t.parsed_alias[0] if t.parsed_alias else ''
    return t