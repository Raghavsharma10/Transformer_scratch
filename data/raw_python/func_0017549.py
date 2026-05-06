def response_minify(self, response):
        """
        minify response html to decrease traffic
        """

        if response.content_type == u'text/html; charset=utf-8':
            endpoint = request.endpoint or ''
            view_func = current_app.view_functions.get(endpoint, None)
            name = (
                '%s.%s' % (view_func.__module__, view_func.__name__)
                if view_func else ''
            )
            if name in self._exempt_routes:
                return response

            response.direct_passthrough = False
            response.set_data(
                self._html_minify.minify(response.get_data(as_text=True))
            )

            return response
        return response