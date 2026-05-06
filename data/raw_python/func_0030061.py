def match(self, path, **kw):
        '''
        path - str (urlencoded)
        '''
        m = self._pattern.match(path)
        if m:
            kwargs = m.groupdict()
            # convert params
            for url_arg_name, value_urlencoded in kwargs.items():
                conv_obj = self._url_params[url_arg_name]
                unicode_value = unquote(value_urlencoded)
                if isinstance(unicode_value, six.binary_type):
                    # XXX ??
                    unicode_value = unicode_value.decode('utf-8', 'replace')
                try:
                    kwargs[url_arg_name] = conv_obj.to_python(unicode_value, **kw)
                except ConvertError as err:
                    logger.debug('ConvertError in parameter "%s" '
                                 'by %r, value "%s"',
                                 url_arg_name,
                                 err.converter.__class__,
                                 err.value)
                    return None, {}
            return m.group(), kwargs
        return None, {}