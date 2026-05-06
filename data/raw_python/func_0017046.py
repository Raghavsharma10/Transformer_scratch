def delete_email_addresses(self, addresses=[]):
        """Delete the email addresses in ``addresses`` from the
        authenticated user's account.

        :param list addresses: (optional), email addresses to be removed
        :returns: bool
        """
        url = self._build_url('user', 'emails')
        return self._boolean(self._delete(url, data=dumps(addresses)),
                             204, 404)