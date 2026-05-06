def get_response(self, request, *args, **kwargs):
        '''Returns the redirect response for this exception.'''
        # normal process
        response = HttpResponseRedirect(self.redirect_to)
        response[REDIRECT_HEADER_KEY] = self.redirect_to
        return response