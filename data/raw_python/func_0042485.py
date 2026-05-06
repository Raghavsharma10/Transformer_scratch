def slugify(text, delim='-'):
    """Generate an ASCII-only slug."""
    result = []
    for word in _punct_re.split((text or '').lower()):
        result.extend(codecs.encode(word, 'ascii', 'replace').split())
    return delim.join([str(r) for r in result])