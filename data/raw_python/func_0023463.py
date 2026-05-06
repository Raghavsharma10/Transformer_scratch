def get(self, path, *args, **kwargs):
        '''GET the provided endpoint'''
        target = self._host.relative(path).utf8
        if not isinstance(target, basestring):
            # on older versions of the `url` library, .utf8 is a method, not a property
            target = target()
        params = kwargs.get('params', {})
        params.update(self._params)
        kwargs['params'] = params
        logger.debug('GET %s with %s, %s', target, args, kwargs)
        return requests.get(target, *args, **kwargs)