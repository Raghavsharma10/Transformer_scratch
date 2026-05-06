def revoke_authorization(self, access_token):
        """Revoke specified authorization for an OAuth application.

        Revoke all authorization tokens created by your application. This will
        only work if you have already called ``set_client_id``.

        :param str access_token: (required), the access_token to revoke
        :returns: bool -- True if successful, False otherwise
        """
        client_id, client_secret = self._session.retrieve_client_credentials()
        url = self._build_url('applications', str(client_id), 'tokens',
                              access_token)
        with self._session.temporary_basic_auth(client_id, client_secret):
            response = self._delete(url, params={'client_id': None,
                                                 'client_secret': None})

        return self._boolean(response, 204, 404)