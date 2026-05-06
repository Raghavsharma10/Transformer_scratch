def open_listing_page(trailing_part_of_url):
    """
    Opens a BBC radio tracklisting page based on trailing part of url.
    Returns a lxml ElementTree derived from that page.

    trailing_part_of_url: a string, like the pid or e.g. pid/segments.inc
    """
    base_url = 'http://www.bbc.co.uk/programmes/'
    print("Opening web page: " + base_url + trailing_part_of_url)

    try:
        html = requests.get(base_url + trailing_part_of_url).text
    except (IOError, NameError):
        print("Error opening web page.")
        print("Check network connection and/or programme id.")
        sys.exit(1)

    try:
        return lxml.html.fromstring(html)
    except lxml.etree.ParserError:
        print("Error trying to parse web page.")
        print("Maybe there's no programme listing?")
        sys.exit(1)