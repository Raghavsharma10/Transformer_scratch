def canonicalize_url(url, keep_params=False, keep_fragments=False):
    """Canonicalize the given url by applying the following procedures:

    # a sort query arguments, first by key, then by value
    # b percent encode paths and query arguments. non-ASCII characters are
    # c percent-encoded using UTF-8 (RFC-3986)
    # d normalize all spaces (in query arguments) '+' (plus symbol)
    # e normalize percent encodings case (%2f -> %2F)
    # f remove query arguments with blank values (unless site in NONCANONIC_SITES)
    # g remove fragments (unless #!)
    # h remove username/password at front of domain
    # i remove port if 80, keep if not
    # k remove query arguments (unless site in USEFUL_QUERY_KEYS)

    The url passed can be a str or unicode, while the url returned is always a
    str.
    """
    if keep_params:
        # Preserve all query params
        parsed = extract(norm(url))
    else:
        # Remove unwanted params
        parsed = extract(url_query_cleaner(normalize(url), parameterlist=config.USEFUL_QUERY_KEYS))

    # Sort params, remove blank if not wanted
    query = urllib.urlencode(sorted(urlparse.parse_qsl(parsed.query, keep_blank_values=keep_params)))
    fragment = getFragment(url, keep_fragments)

    # The following is to remove orphaned '=' from query string params with no values
    query = re.sub(r"=$", "", query.replace("=&", "&"))

    # Reconstruct URL, escaping apart from safe chars
    # See http://stackoverflow.com/questions/2849756/list-of-valid-characters-for-the-fragment-identifier-in-an-url
    # http://stackoverflow.com/questions/4669692/valid-characters-for-directory-part-of-a-url-for-short-links
    safe = "/.-_~!$&'()*+,;=:@"
    newurl = construct(URL(parsed.scheme, '', '', parsed.subdomain, parsed.domain, parsed.tld, parsed.port, quote(parsed.path, safe=safe), query, quote(fragment, safe=safe), ''))
    return newurl.rstrip('/')