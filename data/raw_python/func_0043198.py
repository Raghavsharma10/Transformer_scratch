def _generate_read_callable(name, display_name, arguments, regex, doc, supported):
    """
    Returns a callable which conjures the URL for the resource and GETs a response
    """
    def f(self, *args, **kwargs):
        url = self._generate_url(regex, args)
        if 'params' in kwargs:
            url += "?" + urllib.urlencode(kwargs['params'])
        return self._get_data(url, accept=(kwargs.get('accept')))
    f.__name__ = str('read_%s' % name)
    f.__doc__ = doc
    f._resource_uri = regex
    f._get_args = arguments
    f._put_or_post_args = None 
    f.resource_name = display_name
    f.is_api_call = True
    f.is_supported_api = supported
    return f