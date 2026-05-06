def slugify(s, delimiter='-'):
    """
    Normalize `s` into ASCII and replace non-word characters with `delimiter`.
    """
    s = unicodedata.normalize('NFKD', to_unicode(s)).encode('ascii', 'ignore').decode('ascii')
    return RE_SLUG.sub(delimiter, s).strip(delimiter).lower()