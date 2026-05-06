def munge_author(author):
    """If an author contains an email and a name in it, make sure it is in
    the format: "name (email)"."""
    # this loveliness is from feedparser but was not usable as a function
    if '@' in author:
        emailmatch = re.search(r"(([a-zA-Z0-9\_\-\.\+]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?))(\?subject=\S+)?", author, re.UNICODE)
        if emailmatch:
            email = emailmatch.group(0)
            # probably a better way to do the following, but it passes all the tests
            author = author.replace(email, u'')
            author = author.replace(u'()', u'')
            author = author.replace(u'<>', u'')
            author = author.replace(u'&lt;&gt;', u'')
            author = author.strip()
            if author and (author[0] == u'('):
                author = author[1:]
            if author and (author[-1] == u')'):
                author = author[:-1]
            author = author.strip()
            return '%s (%s)' % (author, email)
    return author