def post_list(self, request, **kwargs):
        """
        (Copied from implementation in
        https://github.com/greenelab/adage-server/blob/master/adage/analyze/api.py)

        Handle an incoming POST as a GET to work around URI length limitations
        """
        # The convert_post_to_VERB() technique is borrowed from
        # resources.py in tastypie source. This helps us to convert the POST
        # to a GET in the proper way internally.
        request.method = 'GET'  # override the incoming POST
        dispatch_request = convert_post_to_VERB(request, 'GET')
        return self.dispatch('list', dispatch_request, **kwargs)