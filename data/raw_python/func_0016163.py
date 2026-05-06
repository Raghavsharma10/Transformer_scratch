def avro_name(url):  # type: (AnyStr) -> AnyStr
    """
    Turn a URL into an Avro-safe name.

    If the URL has no fragment, return this plain URL.

    Extract either the last part of the URL fragment past the slash, otherwise
    the whole fragment.
    """
    frg = urllib.parse.urldefrag(url)[1]
    if frg != '':
        if '/' in frg:
            return frg[frg.rindex('/') + 1:]
        return frg
    return url