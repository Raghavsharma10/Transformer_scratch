def findRelMe(sourceURL):
    """Find all <a /> elements in the given html for a post.

    If any have an href attribute that is rel="me" then include
    it in the result.

    :param sourceURL: the URL for the post we are scanning
    :rtype: dictionary of RelMe references
    """
    r = requests.get(sourceURL)
    result = {'status':  r.status_code,
              'headers': r.headers,
              'history': r.history,
              'content': r.text,
              'relme':   [],
              'url':     sourceURL
              }
    if r.status_code == requests.codes.ok:
        dom = BeautifulSoup(r.text, _html_parser)
        for link in dom.find_all('a', rel='me'):
            rel  = link.get('rel')
            href = link.get('href')
            if rel is not None and href is not None:
                url = urlparse(href)
                if url is not None and url.scheme in ('http', 'https'):
                    result['relme'].append(cleanURL(href))
    return result