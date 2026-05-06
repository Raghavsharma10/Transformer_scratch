def jtype(c):
    """
    Return the a string with the data type of a value, for JSON data
    """
    ct = c['type']
    return ct if ct != 'literal' else '{}, {}'.format(ct, c.get('xml:lang'))