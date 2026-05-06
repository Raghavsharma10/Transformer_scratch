def url(url_pattern, view, kwargs=None, name=None):
    """
    This is replacement for ``django.conf.urls.url`` function.
    This url auto calls ``as_view`` method for Class based views and resolves
    URLPattern objects.

    If ``name`` is not specified it will try to guess it.

    :param url_pattern: string with regular expression or URLPattern
    :param view: function/string/class of the view
    :param kwargs: kwargs that are to be passed to view
    :param name: name of the view, if empty it will be guessed
    """
    # Special handling for included view
    if isinstance(url_pattern, URLPattern) and isinstance(view, tuple):
        url_pattern = url_pattern.for_include()

    if name is None:
        name = resolve_name(view)

    if callable(view) and hasattr(view, 'as_view') and callable(view.as_view):
        view = view.as_view()

    return urls.url(
        regex=url_pattern,
        view=view,
        kwargs=kwargs,
        name=name)