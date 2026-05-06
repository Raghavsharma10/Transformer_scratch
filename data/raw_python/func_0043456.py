def staticbundle(bundle, mimetype=None, **attrs):
    """
    >>> {% staticbundle 'bundlename.css' %}
    >>> {% staticbundle 'bundlename.css' media='screen' %}
    >>> {% staticbundle 'bundlename' mimetype='text/css' %}
    """
    config = getattr(settings, 'STATIC_BUNDLES', {})

    if settings.DEBUG and 'packages' in config and bundle in config['packages']:
        cache_root = os.path.join(settings.STATIC_ROOT, config.get('cache') or DEFAULT_CACHE_DIR)

        bundle_opts = config['packages'][bundle]

        root = os.path.join(cache_root, bundle_opts.get('cwd', ''))

        changed = set()
        src_list = bundle_opts['src']
        is_mapping = isinstance(src_list, dict)

        for src in src_list:
            src_path = os.path.join(bundle_opts.get('cwd', ''), src)
            abs_src = os.path.join(settings.STATIC_ROOT, src_path)

            if is_mapping and not os.path.exists(os.path.join(root, src_list[src])):
                    changed.add(src)

            if abs_src is not None:
                cached_mtime = BUNDLE_CACHE.get(abs_src)
                current_mtime = os.stat(abs_src).st_mtime
                if cached_mtime is None:
                    try:
                        BUNDLE_CACHE[abs_src] = cached_mtime = current_mtime
                    except OSError:
                        cached_mtime = 0
                else:
                    cached_mtime = BUNDLE_CACHE[abs_src]

                if current_mtime != cached_mtime:
                    changed.add(src_path)
                    BUNDLE_CACHE[abs_src] = current_mtime

            elif settings.TEMPLATE_DEBUG:
                raise template.TemplateSyntaxError(
                    "The source file '%s' could not be located." % src_path)

        if changed:
            logger.info('Regenerating %s due to changes: %s', bundle, ' '.join(changed))
            call_command('compilestatic', bundle)

    #     if isinstance(src_list, dict):
    #         src_list = src_list.values()

    src_list = [bundle]

    output = []
    for src_path in src_list:
        url = staticfiles_storage.url(src_path)

        # Some storages backends will yield urls with querystring attached
        path = urlparse.urlparse(url).path
        if path.endswith('.css'):
            mimetype = 'text/css'
        elif path.endswith('.js'):
            mimetype = 'text/javascript'

        output.append(TEMPLATES[mimetype] % dict(
            url=url,
            mimetype=mimetype,
            attrs=' '.join('%s="%s"' % (k, escape(v)) for k, v in attrs.iteritems()),
        ))

    return '\n'.join(output)