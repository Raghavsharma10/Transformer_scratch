def accounts(self):
        """
        Return an account reference
        :param account_id:
        :param accounts_password: The password for decrypting the secret
        :return:
        """
        d = {}

        if False and not self._account_password:
            from ambry.dbexceptions import ConfigurationError
            raise ConfigurationError(
                "Can't access accounts without setting an account password"
                " either in the accounts.password config, or in the AMBRY_ACCOUNT_PASSWORD"
                " env var.")

        for act in self.database.session.query(Account).all():
            if self._account_password:
                act.secret_password = self._account_password
            e = act.dict
            a_id = e['account_id']
            d[a_id] = e

        return d