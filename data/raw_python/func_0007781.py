def _get_request_params(self, **kwargs):
        """Merge shared params and new params."""
        request_params = copy.deepcopy(self._shared_request_params)
        for key, value in iteritems(kwargs):
            if isinstance(value, dict) and key in request_params:
                # ensure we don't lose dict values like headers or cookies
                request_params[key].update(value)
            else:
                request_params[key] = value
        return request_params