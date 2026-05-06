def make_handle_URL(self, handle, indices=None, overwrite=None, other_url=None):
        '''
        Create the URL for a HTTP request (URL + query string) to request
        a specific handle from the Handle Server.

        :param handle: The handle to access.
        :param indices: Optional. A list of integers or strings. Indices of
            the handle record entries to read or write. Defaults to None.
        :param overwrite: Optional. If set, an overwrite flag will be appended
            to the URL ({?,&}overwrite=true or {?,&}overwrite=false). If not set, no
            flag is set, thus the Handle Server's default behaviour will be
            used. Defaults to None.
        :param other_url: Optional. If a different Handle Server URL than the
            one specified in the constructor should be used. Defaults to None.
            If set, it should be set including the URL extension,
            e.g. '/api/handles/'.
        :return: The complete URL, e.g.
         'http://some.handle.server/api/handles/prefix/suffix?index=2&index=6&overwrite=false
        '''
        LOGGER.debug('make_handle_URL...')
        separator = '?'

        if other_url is not None:
            url = other_url
        else:
            url = self.__handle_server_url.strip('/') +'/'+\
                self.__REST_API_url_extension.strip('/')
        url = url.strip('/')+'/'+ handle

        if indices is None:
            indices = []
        if len(indices) > 0:
            for index in indices:
                url = url+separator+'index='+str(index)
                separator = '&'

        if overwrite is not None:
            if overwrite:
                url = url+separator+'overwrite=true'
            else:
                url = url+separator+'overwrite=false'

        return url