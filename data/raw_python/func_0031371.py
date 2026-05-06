def parse_uri(uri):
        ''' parse URI
        '''
        if not uri or uri.find('://') <= 0:
            raise RuntimeError('Incorrect URI definition: {}'.format(uri))
        backend, rest_uri = uri.split('://')
        if backend not in SUPPORTED_BACKENDS:
            raise RuntimeError('Unknown backend: {}'.format(backend))
        database, table = rest_uri.rsplit(':',1)

        return database, table