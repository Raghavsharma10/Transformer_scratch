def hashtag_links(uri, s):
    """ Turns hashtag-like strings into HTML links

        @uri: /uri/ root for the hashtag-like
        @s: the #str string you're looking for |#|hashtags in

        -> #str HTML link |<a href="/uri/hashtag">hashtag</a>|
    """
    for tag, after in hashtag_re.findall(s):
        _uri = '/' + (uri or "").lstrip("/") + quote(tag)
        link = '<a href="{}">#{}</a>{}'.format(_uri.lower(), tag, after)
        s = s.replace('#' + tag, link)
    return s