def prepare_url(self, uri, kwargs):
        """Convert dict for URL params
        """
        params = dict()
        for key in kwargs:
            if key in ('include', 'exclude', 'fields'):
                params.update({
                    key: ','.join(kwargs.get(key))
                })
            elif key in ('search', 'kind'):
                params.update({
                    key: kwargs.get(key)
                })

        if params:
            params = urlencode(params)
            uri = '%s?%s' % (uri, params)

        return uri