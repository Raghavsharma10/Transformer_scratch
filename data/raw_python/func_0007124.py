def _type_string(label, case=None):
    """Shortcut for string like fields"""
    return label, abstractSearch.in_string, lambda s: abstractRender.default(s, case=case), ""