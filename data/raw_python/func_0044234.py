def ftr_process(url=None, content=None, config=None, base_url=None):
    u""" process an URL, or some already fetched content from a given URL.

    :param url: The URL of article to extract. Can be
        ``None``, but only if you provide both ``content`` and
        ``config`` parameters.
    :type url: str, unicode or ``None``

    :param content: the HTML content already downloaded. If given,
        it will be used for extraction, and the ``url`` parameter will
        be used only for site config lookup if ``config`` is not given.
        Please, only ``unicode`` to avoid charset errors.
    :type content: unicode or ``None``

    :param config: if ``None``, it will be looked up from ``url`` with as
        much love and AI as possible. But don't expect too much.
    :type config: a :class:`SiteConfig` instance or ``None``

    :param base_url: reserved parameter, used when fetching multi-pages URLs.
        It will hold the base URL (the first one fetched), and will serve as
        base for fixing non-schemed URLs or query_string-only links to next
        page(s). Please do not set this parameter until you very know what you
        are doing. Default: ``None``.
    :type base_url: str or unicode or None

    :raises:
        - :class:`RuntimeError` in all parameters-incompatible situations.
          Please RFTD carefully, and report strange unicornic edge-cases.
        - :class:`SiteConfigNotFound` if no five-filter site config can
          be found.
        - any raw ``requests.*`` exception, network related, if anything
          goes wrong during url fetching.

    :returns:
        - either a :class:`ContentExtractor` instance with extracted
          (and :attr:`.failures`) attributes set, in case a site config
          could be found.
          When the extractor knows how to handle multiple-pages articles,
          all pages contents will be extracted and cleaned — if relevant —
          and concatenated into the instance :attr:`body` attribute.
          The :attr:`next_page_link` attribute will be a ``list``
          containing all sub-pages links. Note: the first link is the one
          you fed the extractor with ; it will not be repeated in the list.
        - or ``None``, if content was not given and url fetching returned
          a non-OK HTTP code, or if no site config could be found (in that
          particular case, no extraction at all is performed).
    """

    if url is None and content is None and config is None:
        raise RuntimeError('At least one of url or the couple content/config '
                           'argument must be present.')

    if content is not None and url is None and config is None:
        raise RuntimeError('Passing content only will not give any result.')

    if content is None:
        if url is None:
            raise RuntimeError('When content is unset, url must be set.')

        try:
            result = requests_get(url)

            if result.status_code != requests.codes.ok:
                LOGGER.error(u'Wrong status code in return while getting '
                             u'“%s”.', url)
                return None

            # Override before accessing result.text ; see `requests` doc.
            result.encoding = detect_encoding_from_requests_response(result)

            LOGGER.info(u'Downloaded %s bytes as %s text.',
                        len(result.text), result.encoding)

            # result.text is always unicode
            content = result.text

        except:
            LOGGER.error(u'Content could not be fetched from URL %s.', url)
            raise

    if config is None:
        # This can eventually raise SiteConfigNotFound
        config_string, matched_host = ftr_get_config(url)
        config = SiteConfig(site_config_text=config_string, host=matched_host)

    extractor = ContentExtractor(config)

    if base_url is None:
        base_url = url

    if extractor.process(html=content):

        # This is recursive. Yeah.
        if extractor.next_page_link is not None:

            next_page_link = sanitize_next_page_link(extractor.next_page_link,
                                                     base_url)

            next_extractor = ftr_process(url=next_page_link,
                                         base_url=base_url)

            extractor.body += next_extractor.body

            extractor.next_page_link = [next_page_link]

            if next_extractor.next_page_link is not None:
                extractor.next_page_link.extend(next_extractor.next_page_link)

        return extractor

    return None