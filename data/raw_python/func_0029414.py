def register(self):
        """ Register new user by POSTing all required data. """
        user, created = self.Model.create_account(
            self._json_params)

        if not created:
            raise JHTTPConflict('Looks like you already have an account.')

        self.request._user = user
        pk_field = user.pk_field()
        headers = remember(self.request, getattr(user, pk_field))
        return JHTTPOk('Registered', headers=headers)