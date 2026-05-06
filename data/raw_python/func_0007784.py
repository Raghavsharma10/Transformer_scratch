def pre_send(self, request_params):
        """Override this method to modify sent request parameters"""
        for adapter in itervalues(self.adapters):
            adapter.max_retries = request_params.get('max_retries', 0)

        return request_params