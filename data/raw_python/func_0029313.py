def _catch_index_error(self, response):
        """ Catch and raise index errors which are not critical and thus
        not raised by elasticsearch-py.
        """
        code, headers, raw_data = response
        if not raw_data:
            return
        data = json.loads(raw_data)
        if not data or not data.get('errors'):
            return
        try:
            error_dict = data['items'][0]['index']
            message = error_dict['error']
        except (KeyError, IndexError):
            return
        raise exception_response(400, detail=message)