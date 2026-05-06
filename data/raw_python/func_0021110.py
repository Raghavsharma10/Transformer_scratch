def domain_urlize(value):
    """
    Returns an HTML link to the supplied URL, but only using the domain as the
    text. Strips 'www.' from the start of the domain, if present.

    e.g. if `my_url` is 'http://www.example.org/foo/' then:

        {{ my_url|domain_urlize }}

    returns:
        <a href="http://www.example.org/foo/" rel="nofollow">example.org</a>
    """
    parsed_uri = urlparse(value)
    domain = '{uri.netloc}'.format(uri=parsed_uri)

    if domain.startswith('www.'):
        domain = domain[4:]

    return format_html('<a href="{}" rel="nofollow">{}</a>',
            value,
            domain
        )