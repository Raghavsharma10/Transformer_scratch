def findEndpoint(html):
    """Search the given html content for all <link /> elements
    and return any discovered WebMention URL.

    :param html: html content
    :rtype: WebMention URL
    """
    poss_rels = ['webmention', 'http://webmention.org', 'http://webmention.org/', 'https://webmention.org', 'https://webmention.org/']

    # find elements with correct rels and a href value
    all_links = BeautifulSoup(html, _html_parser).find_all(rel=poss_rels, href=True)
    for link in all_links:
        s = link.get('href', None)
        if s is not None:
            return s

    return None