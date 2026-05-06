def authenticate_redirect(self, callback_uri=None,
                              ask_for=["name", "email", "language", "username"]):
        """
        Performs a redirect to the authentication URL for this service.

        After authentication, the service will redirect back to the given
        callback URI.

        We request the given attributes for the authenticated user by
        default (name, email, language, and username). If you don't need
        all those attributes for your app, you can request fewer with
        the |ask_for| keyword argument.
        """
        callback_uri = callback_uri or request.url
        args = self._openid_args(callback_uri, ax_attrs=ask_for)
        return redirect(self._OPENID_ENDPOINT +
                        ("&" if "?" in self._OPENID_ENDPOINT else "?") +
                        urllib.urlencode(args))