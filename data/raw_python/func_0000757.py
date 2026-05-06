def resolve_to_callable(callable_name):
    """ Resolve string :callable_name: to a callable.

    :param callable_name: String representing callable name as registered
        in ramses registry or dotted import path of callable. Can be
        wrapped in double curly brackets, e.g. '{{my_callable}}'.
    """
    from . import registry
    clean_callable_name = callable_name.replace(
        '{{', '').replace('}}', '').strip()
    try:
        return registry.get(clean_callable_name)
    except KeyError:
        try:
            from zope.dottedname.resolve import resolve
            return resolve(clean_callable_name)
        except ImportError:
            raise ImportError(
                'Failed to load callable `{}`'.format(clean_callable_name))