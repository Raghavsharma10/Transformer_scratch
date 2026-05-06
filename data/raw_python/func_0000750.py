def extract_dynamic_part(uri):
    """ Extract dynamic url part from :uri: string.

    :param uri: URI string that may contain dynamic part.
    """
    for part in uri.split('/'):
        part = part.strip()
        if part.startswith('{') and part.endswith('}'):
            return clean_dynamic_uri(part)