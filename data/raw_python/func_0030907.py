def NCBISequenceLinkURL(title, default=None):
    """
    Given a sequence title, like "gi|42768646|gb|AY516849.1| Homo sapiens",
    return the URL of a link to the info page at NCBI.

    title: the sequence title to produce a link URL for.
    default: the value to return if the title cannot be parsed.
    """
    try:
        ref = title.split('|')[3].split('.')[0]
    except IndexError:
        return default
    else:
        return 'http://www.ncbi.nlm.nih.gov/nuccore/%s' % (ref,)