def _parse_header_links(response):
    """
    Parse the links from a Link: header field.

    ..  todo:: Links with the same relation collide at the moment.

    :param bytes value: The header value.

    :rtype: `dict`
    :return: A dictionary of parsed links, keyed by ``rel`` or ``url``.
    """
    values = response.headers.getRawHeaders(b'link', [b''])
    value = b','.join(values).decode('ascii')
    with LOG_HTTP_PARSE_LINKS(raw_link=value) as action:
        links = {}
        replace_chars = u' \'"'
        for val in re.split(u', *<', value):
            try:
                url, params = val.split(u';', 1)
            except ValueError:
                url, params = val, u''

            link = {}
            link[u'url'] = url.strip(u'<> \'"')
            for param in params.split(u';'):
                try:
                    key, value = param.split(u'=')
                except ValueError:
                    break
                link[key.strip(replace_chars)] = value.strip(replace_chars)
            links[link.get(u'rel') or link.get(u'url')] = link
        action.add_success_fields(parsed_links=links)
        return links