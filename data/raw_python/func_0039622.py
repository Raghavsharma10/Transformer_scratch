def unique(transactions):
    """ Remove any duplicate entries. """
    seen = set()
    # TODO: Handle comments
    return [x for x in transactions if not (x in seen or seen.add(x))]