def get_domain_name(url):
    """
    Extract a domain name from the url (without subdomain).

    Args:
        url (str): Url.

    Returns:
        str: Domain name.

    Raises:
        DomainNotMatchedError: If url is wrong.

    Examples:
        >>> get_domain_name('https://vod.tvp.pl/video/')
        'tvp.pl'

        >>> get_domain_name('https://vod')
        Traceback (most recent call last):
        ...
        rtv.exceptions.WrongUrlError: Couldn't match domain name of this url: https://vod

    """
    if not validate_url(url):
        raise WrongUrlError(f'Couldn\'t match domain name of this url: {url}')

    ext = tldextract.extract(url)
    return f'{ext.domain}.{ext.suffix}'