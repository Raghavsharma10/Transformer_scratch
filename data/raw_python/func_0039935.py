def subjects(subjects=None,
             recursive=False,
             include_tables=False,
             lang=DEFAULT_LANGUAGE):
    """List subjects from the subject hierarchy.

    If subjects is not given, the root subjects will be used.

    Returns a generator.
    """
    request = Request('subjects', *subjects,
                      recursive=recursive,
                      includeTables=include_tables,
                      lang=lang)

    return (Subject(subject, lang=lang) for subject in request.json)