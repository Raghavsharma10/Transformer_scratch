def check_authorization(self, response):
        """checks that an authorization call has been made during the request"""
        if not hasattr(request, '_authorized'):
            raise Unauthorized
        elif not request._authorized:
            raise Unauthorized
        return response