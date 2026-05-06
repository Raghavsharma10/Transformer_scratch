def generate_key(url, page_number):
    """
    >>> url_a = 'http://localhost:5009/search?keywords=a'
    >>> generate_key(url_a, 10)
    'http://localhost:5009/search?keywords=a&page=10'
    >>> url_b = 'http://localhost:5009/search?keywords=b&page=1'
    >>> generate_key(url_b, 10)
    'http://localhost:5009/search?keywords=b&page=10'
    """
    index = url.rfind('page')
    if index != -1:
        result = url[0:index]
        result += 'page=%s' % page_number
    else:
        result = url
        result += '&page=%s' % page_number
    return result