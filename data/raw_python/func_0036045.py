def import_opml(url):
    """Import an OPML file locally or from a URL. Uses your text attributes as aliases."""
    # Test if URL given is local, then open, parse out feed urls,
    # add feeds, set text= to aliases and report success, list feeds added
    from bs4 import BeautifulSoup
    try:
        f = file(url).read()
    except IOError:
        f = requests.get(url).text
    soup = BeautifulSoup(f, "xml")
    links = soup.find_all('outline', type="rss" or "pie")
    # This is very slow, might cache this info on add
    for link in links:
        # print link
        add_feed(link['xmlUrl'])
        print("Added " + link['text'])