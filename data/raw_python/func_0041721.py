def alternates(page):
    """
    Return iterable containing players on bench who are available to play,
    where each player is a dict containing:

    - name
    - details
    - opponent
    """
    soup = BeautifulSoup(page)
    try:
        bench = soup.find_all('tr', class_='bench')
        bench_bios = [p.find('div', class_='ysf-player-name') for p in bench]
        names = [p.find('a').text for p in bench_bios]
        details = [p.find('span').text for p in bench_bios]
        opponents = [p.find_all('td', recursive=False)[3].text for p in bench]
        players = [{'name': n, 'details': d, 'opponent': o}
                   for (n, d, o) in zip(names, details, opponents)]
        return [p for p in players if len(p['opponent']) > 0]
    except:
        return None