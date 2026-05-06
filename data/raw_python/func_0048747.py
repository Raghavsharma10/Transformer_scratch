def get_http_method_arg_name(self):
        """
        Return the HTTP function to call and the params/data argument name
        """
        if self.method == 'get':
            arg_name = 'params'
        else:
            arg_name = 'data'
        return getattr(requests, self.method), arg_name