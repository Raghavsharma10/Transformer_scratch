def generic_request(self, method, uri,
                        all_pages=False,
                        data_key=None,
                        no_data=False,
                        do_not_process=False,
                        force_urlencode_data=False,
                        data=None,
                        params=None,
                        files=None,
                        single_item=False):
        """Generic Canvas Request Method."""
        if not uri.startswith('http'):
            uri = self.uri_for(uri)

        if force_urlencode_data is True:
            uri += '?' + urllib.urlencode(data)

        assert method in ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS']

        if method == 'GET':
            response = self.session.get(uri, params=params)
        elif method == 'POST':
            response = self.session.post(uri, data=data, files=files)
        elif method == 'PUT':
            response = self.session.put(uri, data=data)
        elif method == 'DELETE':
            response = self.session.delete(uri, params=params)
        elif method == 'HEAD':
            response = self.session.head(uri, params=params)
        elif method == 'OPTIONS':
            response = self.session.options(uri, params=params)

        response.raise_for_status()

        if do_not_process is True:
            return response

        if no_data:
            return response.status_code

        if all_pages:
            return self.depaginate(response, data_key)

        if single_item:
            r = response.json()
            if data_key:
                return r[data_key]
            else:
                return r

        return response.json()