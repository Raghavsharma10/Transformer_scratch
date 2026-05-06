def get_login_url(self, scope, redirect_uri, state=None,
                      family_names=None, given_names=None, email=None,
                      lang=None, show_login=None):
        """Return a URL for a user to login/register with ORCID.

        Parameters
        ----------
        :param scope: string or iterable of strings
            The scope(s) of the authorization request.
            For example '/authenticate'
        :param redirect_uri: string
            The URI to which the user's browser should be redirected after the
            login.
        :param state: string
            An arbitrary token to prevent CSRF. See the OAuth 2 docs for
            details.
        :param family_names: string
            The user's family name, used to fill the registration form.
        :param given_names: string
            The user's given name, used to fill the registration form.
        :param email: string
            The user's email address, used to fill the sign-in or registration
            form.
        :param lang: string
            The language in which to display the authorization page.
        :param show_login: bool
            Determines whether the log-in or registration form will be shown by
            default.

        Returns
        -------
        :returns: string
            The URL ready to be offered as a link to the user.
        """
        if not isinstance(scope, string_types):
            scope = " ".join(sorted(set(scope)))
        data = [("client_id", self._key), ("scope", scope),
                ("response_type", "code"), ("redirect_uri", redirect_uri)]
        if state:
            data.append(("state", state))
        if family_names:
            data.append(("family_names", family_names.encode("utf-8")))
        if given_names:
            data.append(("given_names", given_names.encode("utf-8")))
        if email:
            data.append(("email", email))
        if lang:
            data.append(("lang", lang))
        if show_login is not None:
            data.append(("show_login", "true" if show_login else "false"))
        return self._login_or_register_endpoint + "?" + urlencode(data)