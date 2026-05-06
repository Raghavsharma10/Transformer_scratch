def javascript_link(src_url, src_path=None, cache_bust=None, content='', extra_attrs=None):
    """ Helper for programmatically building HTML JavaScript source include
    links, with optional cache busting.

    :param src_url:
        Goes into the `src` attribute of the `<script src="...">` tag.

    :param src_path:
        Optional filesystem path to the source file, used when `cache_bust` is
        enabled.

    :param content:
        Optional content of the DOM element. If `None`, then the element is
        self-closed.

    :param cache_bust:
        Optional method to use for cache busting. Can be one of: importtime,
        md5, or mtime. If the value is md5 or mtime, then `src_path` must be
        supplied.


    Example::

        >>> javascript_link('/static/js/core.js')
        u'<script src="/static/js/core.js" type="text/javascript"></script>'
    """
    if cache_bust:
        append_suffix = get_cache_buster(src_path=src_path, method=cache_bust)
        delim = '&' if '?' in src_url else '?'
        src_url += delim + append_suffix

    attrs = {
        'src': src_url,
        'type': 'text/javascript',
    }
    if extra_attrs:
        attrs.update(extra_attrs)

    return tag('script', content=content, attrs=attrs)