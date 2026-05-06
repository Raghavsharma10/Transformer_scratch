def set_who(voevent, date=None, author_ivorn=None):
    """Sets the minimal 'Who' attributes:  date of authoring, AuthorIVORN.

    Args:
        voevent(:class:`Voevent`): Root node of a VOEvent etree.
        date(datetime.datetime): Date of authoring.
            NB Microseconds are ignored, as per the VOEvent spec.
        author_ivorn(str): Short author identifier,
            e.g. ``voevent.4pisky.org/ALARRM``.
            Note that the prefix ``ivo://`` will be prepended internally.

    """
    if author_ivorn is not None:
        voevent.Who.AuthorIVORN = ''.join(('ivo://', author_ivorn))
    if date is not None:
        voevent.Who.Date = date.replace(microsecond=0).isoformat()