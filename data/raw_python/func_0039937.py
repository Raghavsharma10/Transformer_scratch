def tables(subjects=None,
           pastDays=None,
           include_inactive=False,
           lang=DEFAULT_LANGUAGE):
    """Find tables placed under given subjects.
    """
    request = Request('tables',
                      subjects=subjects,
                      pastDays=pastDays,
                      includeInactive=include_inactive,
                      lang=lang)

    return (Table(table, lang=lang) for table in request.json)