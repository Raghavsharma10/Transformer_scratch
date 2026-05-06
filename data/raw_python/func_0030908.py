def NCBISequenceLink(title, default=None):
    """
    Given a sequence title, like "gi|42768646|gb|AY516849.1| Homo sapiens",
    return an HTML A tag dispalying a link to the info page at NCBI.

    title: the sequence title to produce an HTML link for.
    default: the value to return if the title cannot be parsed.
    """
    url = NCBISequenceLinkURL(title)
    if url is None:
        return default
    else:
        return '<a href="%s" target="_blank">%s</a>' % (url, title)