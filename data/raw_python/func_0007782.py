def _sanitize_request_params(self, request_params):
        """Remove keyword arguments not used by `requests`"""
        if 'verify_ssl' in request_params:
            request_params['verify'] = request_params.pop('verify_ssl')
        return dict((key, val) for key, val in request_params.items()
                    if key in self._VALID_REQUEST_ARGS)