def process_response(self, request, response, resource):
        """
        Do response processing
        """
        origin = request.get_header('Origin')
        if not settings.DEBUG:
            if origin in settings.ALLOWED_ORIGINS or not origin:
                response.set_header('Access-Control-Allow-Origin', origin)
            else:
                log.debug("CORS ERROR: %s not allowed, allowed hosts: %s" % (origin,
                                                                             settings.ALLOWED_ORIGINS))
                raise falcon.HTTPForbidden("Denied", "Origin not in ALLOWED_ORIGINS: %s" % origin)
                # response.status = falcon.HTTP_403
        else:
            response.set_header('Access-Control-Allow-Origin', origin or '*')

        response.set_header('Access-Control-Allow-Credentials', "true")
        response.set_header('Access-Control-Allow-Headers', 'Content-Type')
        # This could be overridden in the resource level
        response.set_header('Access-Control-Allow-Methods', 'OPTIONS')