def _type_bool(label,default=False):
    """Shortcut fot boolean like fields"""
    return label, abstractSearch.nothing, abstractRender.boolen, default