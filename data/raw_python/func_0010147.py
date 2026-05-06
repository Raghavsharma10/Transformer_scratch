def clean_url(url, base_url=None):
    """Add base netloc and path to internal URLs and remove www, fragments."""
    parsed_url = urlparse(url)

    fragment = '{url.fragment}'.format(url=parsed_url)
    if fragment:
        url = url.split(fragment)[0]

    # Identify internal URLs and fix their format
    netloc = '{url.netloc}'.format(url=parsed_url)
    if base_url is not None and not netloc:
        parsed_base = urlparse(base_url)
        split_base = '{url.scheme}://{url.netloc}{url.path}/'.format(url=parsed_base)
        url = urljoin(split_base, url)
        netloc = '{url.netloc}'.format(url=urlparse(url))

    if 'www.' in netloc:
        url = url.replace(netloc, netloc.replace('www.', ''))
    return url.rstrip(string.punctuation)