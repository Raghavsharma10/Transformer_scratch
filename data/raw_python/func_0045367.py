def begin(self, request, data):
        """ Try to get Request Token from OAuth Provider and
            redirect user to provider's site for approval.
        """
        request = self.get_request(
                http_url = self.REQUEST_TOKEN_URL,
                parameters = dict(oauth_callback = self.get_callback(request)))
        content = self.load_request(request)
        if not content:
            return redirect('netauth-login')
        request = self.get_request(token = Token.from_string(content), http_url=self.AUTHORIZE_URL)
        return redirect(request.to_url())