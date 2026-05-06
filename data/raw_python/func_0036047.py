def add(url, force=False):
    """Add a atom or RSS feed by url.
    If it doesn't end in .atom or .rss we'll do some guessing."""
    if url[-3:] == 'xml' or url[1][-4:] == 'atom':
        print("Added your feed as %s" % str(add_feed(url)))
    elif is_feed(url):
        print("Added your feed as %s" % str(add_feed(url)))
    elif force:
        print("Added your feed as %s" % str(add_feed(url)))
    else:
        print("Hitman doesn't think that url is a feed; if you're sure it is rerun with --force")