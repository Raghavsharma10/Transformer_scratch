def get_account_details(self, account):
        """
        This method can be used in a number of scenarios:
        1. When it is necessary to very account information
        2. When there's a need to filter transactions by an account id
        3. When account details (e.g. name of account) are needed
        """
        _form = mechanize.HTMLForm(self.SEARCH_MEMBERS_URL, method="POST")
        _form.new_control('text', 'username', {'value': account})
        _form.new_control('text', '_', {'value': ''})

        try:
            r = self.post_url(self.SEARCH_MEMBERS_URL, form=_form)
        except AuthRequiredException:
            self._auth()
            r = self.post_url(self.SEARCH_MEMBERS_URL, form=_form)

        if r:
            # single quoted json parameters are not valid so convert
            # them into double quoted parameters
            _decoded = json.loads(r.replace("'", '"'))
            # we have a double array result so retrieve only what's
            # essential
            if _decoded[0]:
                return _decoded[0][0]

        raise InvalidAccountException