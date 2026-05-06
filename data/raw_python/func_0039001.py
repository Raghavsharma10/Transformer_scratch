def _is_broken_ref(key1, value1, key2, value2):
    """True if this is a broken reference; False otherwise."""
    # A link followed by a string may represent a broken reference
    if key1 != 'Link' or key2 != 'Str':
        return False
    # Assemble the parts
    n = 0 if _PANDOCVERSION < '1.16' else 1

    if isinstance(value1[n][0]['c'], list):
        # Occurs when there is quoted text in an actual link.  This is not
        # a broken link.  See Issue #1.
        return False

    s = value1[n][0]['c'] + value2
    # Return True if this matches the reference regex
    return True if _REF.match(s) else False