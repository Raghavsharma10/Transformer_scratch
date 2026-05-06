def mentions_links(uri, s):
    """ Turns mentions-like strings into HTML links,
        @uri: /uri/ root for the hashtag-like
        @s: the #str string you're looking for |@|mentions in

        -> #str HTML link |<a href="/uri/mention">mention</a>|
    """
    for username, after in mentions_re.findall(s):
        _uri = '/' + (uri or "").lstrip("/") + quote(username)
        link = '<a href="{}">@{}</a>{}'.format(_uri.lower(), username, after)
        s = s.replace('@' + username, link)
    return s