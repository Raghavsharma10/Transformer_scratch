def url_view(url_pattern, name=None, priority=None):
    """
    Decorator for registering functional views.
    Meta decorator syntax has to be used in order to accept arguments.

    This decorator does not really do anything that magical:

    This:
    >>> from urljects import U, url_view
    >>> @url_view(U / 'my_view')
    ... def my_view(request)
    ...     pass

    is equivalent to this:
    >>> def my_view(request)
    ...     pass
    >>> my_view.urljects_view = True
    >>> my_view.url = U / 'my_view'
    >>> my_view.url_name = 'my_view'

    Those view are then supposed to be used with ``view_include`` which will
    register all views that have ``urljects_view`` set to ``True``.

    :param url_pattern: regex or URLPattern or anything passable to url()
    :param name: name of the view, __name__ will be used otherwise.
    :param priority: priority of the view, the lower the better
    """

    def meta_wrapper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.urljects_view = True
        wrapper.url = url_pattern
        wrapper.url_name = name or func.__name__
        wrapper.url_priority = priority

        return wrapper
    return meta_wrapper