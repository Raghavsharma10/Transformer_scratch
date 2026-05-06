def league(page):
    """
    Return the league name
    """
    soup = BeautifulSoup(page)
    try:
        return soup.find('title').text.split(' | ')[0].split(' - ')[0]
    except:
        return None