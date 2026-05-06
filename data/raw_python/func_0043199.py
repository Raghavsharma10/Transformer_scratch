def _generate_create_callable(name, display_name, arguments, regex, doc, supported, post_arguments, is_action):
    """
    Returns a callable which conjures the URL for the resource and POSTs data
    """
    def f(self, *args, **kwargs):
        for key, value in args[-1].items():
            if type(value) == file:
                return self._put_or_post_multipart('POST', self._generate_url(regex, args[:-1]), args[-1])
        return self._put_or_post_json('POST', self._generate_url(regex, args[:-1]), args[-1])
    if is_action:
        f.__name__ = str(name)
    else:
        f.__name__ = str('create_%s' % name)
    f.__doc__ = doc
    f._resource_uri = regex
    f._get_args = arguments
    f._put_or_post_args = post_arguments
    f.resource_name = display_name
    f.is_api_call = True
    f.is_supported_api = supported
    return f