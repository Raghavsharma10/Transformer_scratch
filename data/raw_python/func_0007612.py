def get_sources_by_member(base_url=BASE_URL_API, limit=LIMIT_DEFAULT):
    """
    Function returns which activities each member has joined.

    :param base_url: It is URL: `https://www.openhumans.org/api/public-data`.
    :param limit: It is the limit of data send by one request.
    """
    url = '{}sources-by-member/'.format(base_url)
    page = '{}?{}'.format(url, urlencode({'limit': limit}))
    results = []
    while True:
        data = get_page(page)
        results = results + data['results']
        if data['next']:
            page = data['next']
        else:
            break
    return results