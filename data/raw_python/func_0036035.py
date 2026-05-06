def selective_download(name, oldest, newest):
    """Note: RSS feeds are counted backwards, default newest is 0, the most recent."""
    if six.PY3:
        name = name.encode("utf-8")
    feed = resolve_name(name)
    if six.PY3:
        feed = feed.decode()
    d = feedparser.parse(feed)
    logger.debug(d)
    try:
        d.entries[int(oldest)]
    except IndexError:
        print("Error feed does not contain this many items.")
        print("Hitman thinks there are %d items in this feed." % len(d.entries))
        return
    for url in [q.enclosures[0]['href'] for q in d.entries[int(newest):int(oldest)]]:
        # iterate over urls in feed from newest to oldest feed items.
        url = str(url)
        with Database("downloads") as db:
            if url.split('/')[-1] not in db:
                # download(url, name, feed)
                with Database("settings") as settings:
                    if 'dl' in settings:
                        dl_dir = settings['dl']
                    else:
                        dl_dir = os.path.join(os.path.expanduser("~"), "Downloads")
                requests_get(url, dl_dir)