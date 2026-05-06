def get_authenticated_user(self, callback):
        """Fetches the authenticated user data upon redirect.

        This method should be called by the handler that receives the
        redirect from the authenticate_redirect() or authorize_redirect()
        methods.
        """
        # Verify the OpenID response via direct request to the OP
        args = dict((k, v) for k, v in request.args.items())
        args["openid.mode"] = u"check_authentication"

        r = requests.post(self._OPENID_ENDPOINT, data=args)
        return self._on_authentication_verified(callback, r)