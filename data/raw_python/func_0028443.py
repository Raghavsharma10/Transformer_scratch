def summary_extra(self):
        """
        Build the extra data for the summary logger.
        """
        out = {
            'errno': 0,
            'agent': request.headers.get('User-Agent', ''),
            'lang': request.headers.get('Accept-Language', ''),
            'method': request.method,
            'path': request.path,
        }

        # set the uid value to the current user ID
        user_id = self.user_id()
        if user_id is None:
            user_id = ''
        out['uid'] = user_id

        # the rid value to the current request ID
        request_id = g.get('_request_id', None)
        if request_id is not None:
            out['rid'] = request_id

        # and the t value to the time it took to render
        start_timestamp = g.get('_start_timestamp', None)
        if start_timestamp is not None:
            # Duration of request, in milliseconds.
            out['t'] = int(1000 * (time.time() - start_timestamp))

        return out