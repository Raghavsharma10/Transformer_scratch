def _reinsert_root_tag_prefix(v):
    """
    Returns namespace prefix to root tag, if it had one.
    """
    if hasattr(v, 'original_prefix'):
        original_prefix = v.original_prefix
        del v.original_prefix
        v.tag = ''.join(('{', v.nsmap[original_prefix], '}VOEvent'))
    return