def get_realtime_urls(admin_view_func=lambda x: x):
    """
    Get the URL for real-time widgets.

    Args:
        admin_view_func (callable): an admin_view method from an AdminSite
            instance. By default: identity.

    Returns:
        list: the list of the real-time URLs as django's ``url()``.
    """
    from .widgets import REALTIME_WIDGETS
    return [url(w.url_regex, admin_view_func(w.as_view()), name=w.url_name)
            for w in REALTIME_WIDGETS]