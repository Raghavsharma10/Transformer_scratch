def ftr_string_to_instance(config_string):
    """ Return a :class:`SiteConfig` built from a ``config_string``.

    Simple syntax errors are just plainly ignored, and logged as warnings.

    :param config_string: a full site config file, raw-loaded from storage
        with something like
        ``config_string = open('path/to/site/config.txt', 'r').read()``.
    :type config_string: str or unicode

    :returns: a :class:`SiteConfig` instance.
    :raises: :class:`InvalidSiteConfig` in case of an unrecoverable error.

    .. note:: See the source code for supported directives names.
    """

    config = SiteConfig()

    for line_number, line_content in enumerate(
            config_string.strip().split(u'\n'), start=1):

        line_content = line_content.strip()

        # Skip empty lines & comments.
        if not line_content or line_content.startswith(u'#'):
            continue

        try:
            key, value = [
                x.strip() for x in line_content.strip().split(u':', 1)
            ]

        except:
            LOGGER.warning(u'Unrecognized syntax “%s” on line #%s.',
                           line_content, line_number)
            continue

        # handle some very rare title()d directives.
        key = key.lower()

        if not key or (not value and key != 'replace_string'):
            LOGGER.warning(u'Empty key or value in “%s” on line #%s.',
                           line_content, line_number)
            continue

        # Commands for which we accept multiple statements.
        elif key in (
            'title', 'body', 'author', 'date',
            'strip', 'strip_id_or_class', 'strip_image_src',
            'single_page_link', 'single_page_link_in_feed',
            'next_page_link',
            'http_header',

            'find_string',
            'replace_string',

            'test_url',
            'test_contains',
            'test_title',
            'test_date',
            'test_author',
            'test_language',
        ):

            if key.endswith(u'_string'):
                # Append to list. Duplicites are allowed.
                getattr(config, key).append(value)

            else:
                # Add to set, preserving order but squashing duplicates.
                getattr(config, key).add(value)

        # Single statement commands that evaluate to True or False.
        elif key in ('tidy', 'prune', 'autodetect_on_failure', ):

            if value.lower() in ('no', 'false', '0', ):
                setattr(config, key, False)

            else:
                setattr(config, key, bool(value))

        # Single statement commands stored as strings.
        elif key in ('parser', ):
            setattr(config, key, value)

        # The “replace_string(………): replace_value” one-liner syntax.
        elif key.startswith('replace_string(') and key.endswith(')'):
            # These 2 are lists, not sets.
            config.find_string.append(key[15:-1])
            config.replace_string.append(value)

        else:
            LOGGER.warning(u'Unsupported directive “%s” on line #%s.',
                           line_content, line_number)

    find_count = len(config.find_string)
    replace_count = len(config.replace_string)

    if find_count != replace_count:
        raise InvalidSiteConfig(u'find_string and remplace_string do not '
                                u'correspond ({0} != {1})'.format(
                                    find_count, replace_count))

    return config