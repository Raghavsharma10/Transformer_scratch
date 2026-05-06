def seven_zip(archive, items, self_extracting=False):
    """Create a 7z archive."""
    if not isinstance(items, (list, tuple)):
        items = [items]
    if self_extracting:
        return er(_get_sz(), "a", "-ssw", "-sfx", archive, *items)
    else:
        return er(_get_sz(), "a", "-ssw", archive, *items)