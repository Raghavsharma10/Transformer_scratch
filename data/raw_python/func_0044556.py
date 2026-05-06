def authorization_header_parameters(self):
        """
        The parameters from the Authorization header (only).  If the
        Authorization header is not present or is not an AWS SigV4 header, an
        AttributeError exception is raised.
        """
        result = getattr(self, "_authorization_header_parameters", None)
        if result is None:
            auth = self.headers.get(_authorization)
            if auth is None:
                raise AttributeError("Authorization header is not present")
            
            if not auth.startswith(AWS4_HMAC_SHA256 + " "):
                raise AttributeError("Authorization header is not AWS SigV4")

            result = {}
            for parameter in auth[len(AWS4_HMAC_SHA256)+1:].split(","):
                parameter = parameter.strip()
                try:
                    key, value = parameter.split("=", 1)
                except ValueError:
                    raise AttributeError(
                        "Invalid Authorization header: missing '='")

                if key in result:
                    raise AttributeError(
                        "Invalid Authorization header: duplicate key %r" % key)
                
                result[key] = value
            
            self._authorization_header_parameters = result
        return result