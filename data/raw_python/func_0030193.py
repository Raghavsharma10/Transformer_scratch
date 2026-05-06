def sync_accounts(self, accounts_data, clear = False, password=None, cb = None):
        """
        Load all of the accounts from the account section of the config
        into the database.

        :param accounts_data:
        :param password:
        :return:
        """

        # Map common values into the accounts records

        all_accounts = self.accounts

        kmap = Account.prop_map()

        for account_id, values in accounts_data.items():

            if not isinstance(values, dict):
                continue

            d = {}

            a = self.library.find_or_new_account(account_id)
            a.secret_password = password or self.password

            for k, v in values.items():
                if k in ('id',):
                    continue
                try:
                    if kmap[k] == 'secret' and v:
                        a.encrypt_secret(v)
                    else:
                        setattr(a, kmap[k], v)
                except KeyError:
                    d[k] = v

            a.data = d

            if values.get('service') == 's3':
                a.url = 's3://{}'.format(a.account_id)

            if cb:
                cb('Loaded account: {}'.format(a.account_id))

            self.database.session.commit()