def match_blocks(hash_func, old_children, new_children):
    """Use difflib to find matching blocks."""
    sm = difflib.SequenceMatcher(
        _is_junk,
        a=[hash_func(c) for c in old_children],
        b=[hash_func(c) for c in new_children],
    )
    return sm