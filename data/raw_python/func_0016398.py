def build_url(self, path, params=None):
        '''
        Constructs the url for a cheddar API resource
        '''
        url = u'%s/%s/productCode/%s' % (
            self.endpoint,
            path,
            self.product_code,
        )
        if params:
            for key, value in params.items():
                url = u'%s/%s/%s' % (url, key, value)

        return url