def get_soup(page=''):
    """
    Returns a bs4 object of the page requested
    """
    content = requests.get('%s/%s' % (BASE_URL, page)).text
    return BeautifulSoup(content)