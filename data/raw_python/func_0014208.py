def get_response(self, request):
        '''Returns the redirect response for this exception.'''
        # the redirect key is already placed in the response by HttpResponseJavascriptRedirect
        return HttpResponseJavascriptRedirect(self.redirect_to, *self.args, **self.kwargs)