def date(page):
    """
    Return the date, nicely-formatted
    """
    soup = BeautifulSoup(page)
    try:
        page_date = soup.find('input', attrs={'name': 'date'})['value']
        parsed_date = datetime.strptime(page_date, '%Y-%m-%d')
        return parsed_date.strftime('%a, %b %d, %Y')
    except:
        return None