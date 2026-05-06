def get_vary_headers(self, request, response):
        """
        Hook for patching the vary header
        """

        headers = []
        accessed = False
        try:
            accessed = request.session.accessed
        except AttributeError:
            pass

        if accessed:
            headers.append("Cookie")
        return headers