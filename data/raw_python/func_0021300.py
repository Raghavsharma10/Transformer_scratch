def request(self, method, url, params=None, **kwargs):
        """Perform a request, or return a cached response if available."""
        params_key = tuple(params.items()) if params else ()
        if method.upper() == "GET":
            if (url, params_key) in self.get_cache:
                print("Returning cached response for:", method, url, params)
                return self.get_cache[(url, params_key)]
        result = super().request(method, url, params, **kwargs)
        if method.upper() == "GET":
            self.get_cache[(url, params_key)] = result
            print("Adding entry to the cache:", method, url, params)
        return result