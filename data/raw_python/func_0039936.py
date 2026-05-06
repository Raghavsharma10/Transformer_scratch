def tableinfo(tableid, lang=DEFAULT_LANGUAGE):
    """Fetch metadata for statbank table

    Metadata includes information about variables,
    which can be used when extracting data.
    """
    request = Request('tableinfo', tableid, lang=lang)

    return Tableinfo(request.json, lang=lang)