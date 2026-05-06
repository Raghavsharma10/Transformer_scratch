def write_listing_to_textfile(textfile, tracklisting):
    """Write tracklisting to a text file."""
    with codecs.open(textfile, 'wb', 'utf-8') as text:
        text.write(tracklisting)