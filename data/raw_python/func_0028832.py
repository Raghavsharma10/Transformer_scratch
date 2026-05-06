def set_response_headers(self, response: HttpResponse) -> HttpResponse:
        """
        Appends default headers to every response returned by the API
        :param response HttpResponse
        :rtype: HttpResponse
        """
        public_name = os.environ.get('SERVER_PUBLIC_NAME')
        response_headers = {
            'access-control-allow-headers': self.app.allowed_headers,
            'access-control-allow-methods': self.app.allowed_methods,
            'access-control-allow-origin': self.app.allowed_origins,
            'access-control-allow-credentials': True,
            'www-authenticate': "Bearer",
            'server-public-name': public_name if public_name else "No one",
            'user-info': "Rinzler Framework rulez!"
        }

        response_headers.update(self.app.default_headers)

        for key in response_headers:
            response[key] = response_headers[key]

        status = response.status_code
        if status != 404:
            self.app.log.info("< {0}".format(status))

        return response