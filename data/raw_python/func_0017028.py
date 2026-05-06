def pubsubhubbub(self, mode, topic, callback, secret=''):
        """Create/update a pubsubhubbub hook.

        :param str mode: (required), accepted values: ('subscribe',
            'unsubscribe')
        :param str topic: (required), form:
            https://github.com/:user/:repo/events/:event
        :param str callback: (required), the URI that receives the updates
        :param str secret: (optional), shared secret key that generates a
            SHA1 HMAC of the payload content.
        :returns: bool
        """
        from re import match
        m = match('https?://[\w\d\-\.\:]+/\w+/[\w\._-]+/events/\w+', topic)
        status = False
        if mode and topic and callback and m:
            data = [('hub.mode', mode), ('hub.topic', topic),
                    ('hub.callback', callback)]
            if secret:
                data.append(('hub.secret', secret))
            url = self._build_url('hub')
            # This is not JSON data. It is meant to be form data
            # application/x-www-form-urlencoded works fine here, no need for
            # multipart/form-data
            status = self._boolean(self._post(url, data=data, json=False), 204,
                                   404)
        return status