def create_html_from_fragment(tag):
    """
    Creates full html tree from a fragment. Assumes that tag should be wrapped in a body and is currently not

    Args:
        tag: a bs4.element.Tag

    Returns:"
        bs4.element.Tag: A bs4 tag representing a full html document
    """

    try:
        assert isinstance(tag, bs4.element.Tag)
    except AssertionError:
        raise TypeError
    try:
        assert tag.find_all('body') == []
    except AssertionError:
        raise ValueError

    soup = BeautifulSoup('<html><head></head><body></body></html>', 'html.parser')
    soup.body.append(tag)
    return soup