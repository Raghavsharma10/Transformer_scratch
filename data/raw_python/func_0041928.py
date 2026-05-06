def fetcher(date=datetime.today(), url_pattern=URL_PATTERN):
    """
    Fetch json data from n.pl

    Args:
        date (date) - default today
        url_patter (string) - default URL_PATTERN

    Returns:
        dict - data from api
    """
    api_url = url_pattern % date.strftime('%Y-%m-%d')

    headers = {'Referer': 'http://n.pl/program-tv'}
    raw_result = requests.get(api_url, headers=headers).json()
    return raw_result