def writeDetails(accept, readId, taxonomy, fp):
    """
    Write read and taxonomy details.

    @param accept: A C{bool} indicating whether the read was accepted,
        according to its taxonomy.
    @param readId: The C{str} id of the read.
    @taxonomy: A C{list} of taxonomy C{str} levels.
    @fp: An open file pointer to write to.
    """
    fp.write('%s %s\n       %s\n\n' % (
        'MATCH:' if accept else 'MISS: ', readId,
        ' | '.join(taxonomy) if taxonomy else 'No taxonomy found.'))