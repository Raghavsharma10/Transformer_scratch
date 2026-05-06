def _headers(self, others={}):
        """Return the default headers and others as necessary"""
        headers = {
            'Content-Type': 'application/json'
        }

        for p in others.keys():
            headers[p] = others[p]
        return headers