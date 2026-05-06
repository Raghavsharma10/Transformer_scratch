def export_opml():
    """Export an OPML feed list"""
    with Database("feeds") as feeds:
        # Thanks to the canto project- used under the GPL
        print("""<opml version="1.0">""")
        print("""<body>""")
        # Accurate but slow.
        for name in list(feeds.keys()):
            kind = feedparser.parse(feeds[name]).version
            if kind[:4] == 'atom':
                t = 'pie'
            elif kind[:3] == 'rss':
                t = 'rss'
            print("""\t<outline text="%s" xmlUrl="%s" type="%s" />""" % (name, feeds[name], t))
        print("""</body>""")
        print("""</opml>""")