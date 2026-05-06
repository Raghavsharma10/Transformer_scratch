def pre_request(self):
        """
        List of pre-request methods from registered middleware.
        """
        middleware = sort_by_priority(self)
        return tuple(m.pre_request for m in middleware if hasattr(m, 'pre_request'))