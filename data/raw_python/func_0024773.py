def create_body(action, params):
        """Create http body for rest request."""
        body = {}
        body['action'] = action
        if params is not None:
            body['params'] = params
        return body