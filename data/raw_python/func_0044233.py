def sanitize_next_page_link(next_page_link, base_url):
    """ Convert relative links or query_string only links to absolute URLs. """

    if not next_page_link.startswith(u'http'):
        if next_page_link.startswith(u'?'):
            # We have some "?current_page=2" scheme.
            next_page_link = base_url + next_page_link

        if next_page_link.startswith(u'/'):
            # We have a server-relative path.

            try:
                proto, host_and_port, remaining = split_url(base_url)

            except:
                LOGGER.error(u'Could not split “%s” to get schema/host parts, '
                             u'next_page_link “%s” will be unusable.',
                             base_url, next_page_link)

            else:
                next_page_link = '{0}://{1}{2}'.format(proto,
                                                       host_and_port,
                                                       next_page_link)
        else:
            LOGGER.warning(u'Unimplemented scheme in '
                           u'next_page_link %s',
                           next_page_link)

    return next_page_link