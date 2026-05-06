def team(page):
    """
    Return the team name
    """
    soup = BeautifulSoup(page)
    try:
        return soup.find('title').text.split(' | ')[0].split(' - ')[1]
    except:
        return None