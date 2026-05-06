def get_uwsgi_stub_attrs_diff():
    """Returns attributes difference two elements tuple between
    real uwsgi module and its stub.

    Might be of use while describing in stub new uwsgi functions.

    :return: (uwsgi_only_attrs, stub_only_attrs)

    :rtype: tuple
    """

    try:
        import uwsgi

    except ImportError:
        from uwsgiconf.exceptions import UwsgiconfException

        raise UwsgiconfException(
            '`uwsgi` module is unavailable. Calling `get_attrs_diff` in such environment makes no sense.')

    from . import uwsgi_stub

    def get_attrs(src):
        return set(attr for attr in dir(src) if not attr.startswith('_'))

    attrs_uwsgi = get_attrs(uwsgi)
    attrs_stub = get_attrs(uwsgi_stub)

    from_uwsgi = sorted(attrs_uwsgi.difference(attrs_stub))
    from_stub = sorted(attrs_stub.difference(attrs_uwsgi))

    return from_uwsgi, from_stub