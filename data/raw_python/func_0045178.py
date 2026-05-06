def generate_output(listing, title, date):
    """
    Returns a string containing a full tracklisting.

    listing: list of (artist(s), track, record label) tuples
    title: programme title
    date: programme date
    """
    listing_string = '{0}\n{1}\n\n'.format(title, date)
    for entry in listing:
        listing_string += '\n'.join(entry) + '\n***\n'
    return listing_string