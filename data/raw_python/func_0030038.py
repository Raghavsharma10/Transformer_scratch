def account(self):
        """Return an account record, based on the host in the url"""
        from ambry.util import parse_url_to_dict

        d = parse_url_to_dict(self.url)

        return self._bundle.library.account(d['netloc'])