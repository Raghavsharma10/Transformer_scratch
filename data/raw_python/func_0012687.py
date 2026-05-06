def findMentions(sourceURL, targetURL=None, exclude_domains=[], content=None, test_urls=True, headers={}, timeout=None):
    """Find all <a /> elements in the given html for a post. Only scan html element matching all criteria in look_in.

    optionally the content to be scanned can be given as an argument.

    If any have an href attribute that is not from the
    one of the items in exclude_domains, append it to our lists.

    :param sourceURL: the URL for the post we are scanning
    :param exclude_domains: a list of domains to exclude from the search
    :type exclude_domains: list
    :param content: the content to be scanned for mentions
    :param look_in: dictionary with name, id and class_. only element matching all of these will be scanned
    :param test_urls: optional flag to test URLs for validation
    :param headers: optional headers to send with any web requests
    :type headers: dict
    :param timeout: optional timeout for web requests
    :type timeout float
    :rtype: dictionary of Mentions
    """

    __doc__ = None

    if test_urls:
        URLValidator(message='invalid source URL')(sourceURL)

    if content:
        result = {'status':   requests.codes.ok,
                  'headers':  None,
                  }
    else:
        r = requests.get(sourceURL, verify=True, headers=headers, timeout=timeout)
        result = {'status':   r.status_code,
                  'headers':  r.headers
                  }
        # Check for character encodings and use 'correct' data
        if 'charset' in r.headers.get('content-type', ''):
            content = r.text
        else:
            content = r.content

    result.update({'refs': set(), 'post-url': sourceURL})

    if result['status'] == requests.codes.ok:
        # Allow passing BS doc as content
        if isinstance(content, BeautifulSoup):
            __doc__ = content
            # result.update({'content': unicode(__doc__)})
            result.update({'content': str(__doc__)})
        else:
            __doc__ = BeautifulSoup(content, _html_parser)
            result.update({'content': content})

        # try to find first h-entry else use full document
        entry = __doc__.find(class_="h-entry") or __doc__

        # Allow finding particular URL
        if targetURL:
            # find only targetURL
            all_links = entry.find_all('a', href=targetURL)
        else:
            # find all links with a href
            all_links = entry.find_all('a', href=True)
        for link in all_links:
            href = link.get('href', None)
            if href:
                url = urlparse(href)

                if url.scheme in ('http', 'https'):
                    if url.hostname and url.hostname not in exclude_domains:
                        result['refs'].add(href)
    return result