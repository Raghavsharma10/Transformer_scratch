def add_feed(url):
    """add to db"""
    with Database("feeds") as db:
        title = feedparser.parse(url).feed.title
        name = str(title)
        db[name] = url
        return name