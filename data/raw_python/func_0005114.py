def cors_setup(self, request):
        """
        Sets up the CORS headers response based on the settings used for the API.

        :param request: <pyramid.request.Request>
        """
        def cors_headers(request, response):
            if request.method.lower() == 'options':
                response.headers.update({
                    '-'.join([p.capitalize() for p in k.split('_')]): v
                    for k, v in self.cors_options.items()
                })
            else:
                origin = self.cors_options.get('access_control_allow_origin', '*')
                expose_headers = self.cors_options.get('access_control_expose_headers', '')
                response.headers['Access-Control-Allow-Origin'] = origin
                if expose_headers:
                    response.headers['Access-Control-Expose-Headers'] = expose_headers

        # setup the CORS supported response
        request.add_response_callback(cors_headers)