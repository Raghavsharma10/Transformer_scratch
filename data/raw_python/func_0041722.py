def start_active_players_path(page):
    """
    Return the path in the "Start Active Players" button
    """
    soup = BeautifulSoup(page)
    try:
        return soup.find('a', href=True, text='Start Active Players')['href']
    except:
        return None