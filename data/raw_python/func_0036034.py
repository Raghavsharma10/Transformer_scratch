def put_a_hit_out(name):
    """Download a feed's most recent enclosure that we don't have"""

    feed = resolve_name(name)
    if six.PY3:
        feed = feed.decode()
    d = feedparser.parse(feed)
    # logger.info(d)
    # logger.info(feed)
    print(d['feed']['title'])
    if d.entries[0].enclosures:
        with Database("settings") as s:
            if 'verbose' in s:
                print(d.entries[0].enclosures[0])

        # print d.feed.updated_parsed
        # Doesn't work everywhere, may nest in try or
        # use .headers['last-modified']
        url = str(d.entries[0].enclosures[0]['href'])
        with Database("downloads") as db:
            if url.split('/')[-1] not in db:
                with Database("settings") as settings:
                    if 'dl' in settings:
                        dl_dir = settings['dl']
                    else:
                        dl_dir = os.path.join(os.path.expanduser("~"), "Downloads")
                requests_get(url, dl_dir)
                db[url.split('/')[-1]] = json.dumps({'url': url, 'date': time.ctime(), 'feed': feed})
                growl("Mission Complete: %s downloaded" % d.feed.title)
                print("Mission Complete: %s downloaded" % d.feed.title)
            else:
                growl("Mission Aborted: %s already downloaded" % d.feed.title)
                print("Mission Aborted: %s already downloaded" % d.feed.title)