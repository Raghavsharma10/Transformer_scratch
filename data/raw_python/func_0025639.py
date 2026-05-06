def get_callback(self, renderer_context):
        """
        Determine the name of the callback to wrap around the json output.
        """
        request = renderer_context.get('request', None)
        params = request and get_query_params(request) or {}
        return params.get(self.callback_parameter, self.default_callback)