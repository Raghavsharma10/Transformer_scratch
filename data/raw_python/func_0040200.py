def drop_it(title, filters, blacklist):
    """
    The found torrents should be in filters list and shouldn't be in blacklist.
    """
    title = title.lower()
    matched = False
    for f in filters:
        if re.match(f, title):
            matched = True
    if not matched:
        return True
    for b in blacklist:
        if re.match(b, title):
            return True
    return False