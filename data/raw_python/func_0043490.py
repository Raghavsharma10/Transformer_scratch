def process_response(self, request, response):
        """
        Adds WWW-Authenticate: Basic headers to 401 responses, and rewrites
        redirects the login page to be 401 responses if it's a non-browser
        agent.
        """
        process = False

        # Don't do anything for unsecure requests, unless DEBUG is on
        if not self.allow_http and not request.is_secure():
            return response

        if response.status_code == UNAUTHORIZED:
            pass
        elif response.status_code == FOUND:
            location = urllib_parse.urlparse(response['Location'])
            if location.path != settings.LOGIN_URL:
                # If it wasn't a redirect to the login page, we don't touch it.
                return response
            elif not self.is_agent_a_robot(request):
                # We don't touch requests made in order to be shown to humans.
                return response

        realm = getattr(settings, 'BASIC_AUTH_REALM', request.META.get('HTTP_HOST', 'restricted'))
        
        if response.status_code == FOUND:
            response = self.unauthorized_view(request)

        authenticate = response.get('WWW-Authenticate', None)
        if authenticate:
            authenticate = 'Basic realm="%s", %s' % (realm, authenticate)
        else:
            authenticate = 'Basic realm="%s"' % realm
        response['WWW-Authenticate'] = authenticate

        return response