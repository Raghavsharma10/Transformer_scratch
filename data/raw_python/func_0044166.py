def ftr_get_config(website_url, exact_host_match=False):
    """ Download the Five Filters config from centralized repositories.

    Repositories can be local if you need to override siteconfigs.

    The first entry found is returned. If no configuration is found,
    `None` is returned. If :mod:`cacheops` is installed, the result will
    be cached with a default expiration delay of 3 days.

    :param exact_host_match: If ``False`` (default), we will look for
        wildcard config matches. For example if host is
        ``www.test.example.org``, we will try looking up
        ``test.example.org`` and ``example.org``.
    :param exact_host_match: bool

    :param website_url: either a full web URI (eg.
        ``http://www.website.com:PORT/path/to/a/page.html``) or simply
        a domain name (eg. ``www.website.com``). In case of a domain name,
        no check is performed yet, be careful of what you pass.
    :type website_url: str or unicode

    :returns: tuple -- the loaded site config (as unicode string) and
        the hostname matched (unicode string too).
    :raises: :class:`SiteConfigNotFound` if no config could be found.

    .. note:: Whatever ``exact_host_match`` value is, the ``www`` part is
        always removed from the URL or domain name.

    .. todo:: there is currently no merging/cascading of site configs. In
        the original Five Filters implementation, primary and secondary
        configurations were merged. We could eventually re-implement this
        part if needed by someone. PRs welcome as always.
    """

    def check_requests_result(result):
        return (
            u'text/plain' in result.headers.get('content-type')
            and u'<!DOCTYPE html>' not in result.text
            and u'<html ' not in result.text
            and u'</html>' not in result.text
        )

    repositories = [
        x.strip() for x in os.environ.get(
            'PYTHON_FTR_REPOSITORIES',
            os.path.expandvars(u'${HOME}/sources/ftr-site-config') + u' '
            + u'https://raw.githubusercontent.com/1flow/ftr-site-config/master/ '  # NOQA
            + u'https://raw.githubusercontent.com/fivefilters/ftr-site-config/master/'  # NOQA
        ).split() if x.strip() != u'']

    try:
        proto, host_and_port, remaining = split_url(website_url)

    except:
        host_and_port = website_url

    host_domain_parts = host_and_port.split(u'.')

    # we don't store / use the “www.” part of domain name in siteconfig.
    if host_domain_parts[0] == u'www':
        host_domain_parts = host_domain_parts[1:]

    if exact_host_match:
        domain_names = [u'.'.join(host_domain_parts)]

    else:
        domain_names = [
            u'.'.join(host_domain_parts[-i:])
            for i in reversed(range(2, len(host_domain_parts) + 1))
        ]

    LOGGER.debug(u'Gathering configurations for domains %s from %s.',
                 domain_names, repositories)

    for repository in repositories:
        # try, in turn:
        #   website.ext.txt
        #   .website.ext.txt

        for domain_name in domain_names:

            skip_repository = False

            for txt_siteconfig_name in (
                u'{0}.txt'.format(domain_name),
                u'.{0}.txt'.format(domain_name),
            ):
                if repository.startswith('http'):
                    siteconfig_url = repository + txt_siteconfig_name

                    result = requests.get(siteconfig_url)

                    if result.status_code == requests.codes.ok:
                        if not check_requests_result(result):
                            LOGGER.error(u'“%s” repository URL does not '
                                         u'return text/plain results.',
                                         repository)
                            skip_repository = True
                            break

                        LOGGER.info(u'Using remote siteconfig for domain '
                                    u'%s from %s.', domain_name,
                                    siteconfig_url, extra={
                                        'siteconfig': domain_name})
                        return result.text, txt_siteconfig_name[:-4]

                else:
                    filename = os.path.join(repository, txt_siteconfig_name)

                    if os.path.exists(filename):
                        LOGGER.info(u'Using local siteconfig for domain '
                                    u'%s from %s.', domain_name,
                                    filename, extra={
                                        'siteconfig': domain_name})

                        with codecs.open(filename, 'rb', encoding='utf8') as f:
                            return f.read(), txt_siteconfig_name[:-4]

                if skip_repository:
                    break

            if skip_repository:
                break

    raise SiteConfigNotFound(
        u'No configuration found for domains {0} in repositories {1}'.format(
            u', '.join(domain_names), u', '.join(repositories)
        )
    )