def account(self, url):
        """
        Return accounts references for the given account id.
        :param account_id:
        :param accounts_password: The password for decrypting the secret
        :return:
        """
        from sqlalchemy.orm.exc import NoResultFound
        from ambry.orm.exc import NotFoundError
        from ambry.util import parse_url_to_dict
        from ambry.orm import Account

        pd = parse_url_to_dict(url)

        # Old method of storing account information.
        try:
            act = self.database.session.query(Account).filter(Account.account_id == pd['netloc']).one()
            act.secret_password = self._account_password
            return act
        except NoResultFound:
            pass

        # Try the remotes.
        for r in self.remotes:
            if url.startswith(r.url):
                return r


        raise NotFoundError("Did not find account for url: '{}' ".format(url))