def get_transactions(self, **kwargs):
        """
        This method optionally takes the following extra
        keyword arguments:
        to_date: a datetime object representing the date the filter should end with
        from_date: a datetime object representing the date the filter should start from
        txn_ref: the transaction reference of a particular transaction
        from_account_id: the account id for the account to filter transactions by (you will
        need to get this information from `get_account_details` method)
        If you specify txn_ref, then it's not necessary to specify to_date and from_date.
        """
        kw_map = {
            'to_date': 'query(period).end',
            'from_account_id': 'query(member)',
            'from_date': 'query(period).begin',
            'txn_ref': 'query(transactionNumber)'}

        if not self.TRANSACTIONS_FORM:
            try:
                self.get_url(self.TRANSACTIONS_URL)
            except AuthRequiredException:
                self._auth()
                self.get_url(self.TRANSACTIONS_URL)
            self.br.select_form("accountHistoryForm")
            self.br.form.method = 'POST'
            self.br.form.action = self.TRANSACTIONS_EXPORT_URL
            self.TRANSACTIONS_FORM = self.br.form
            _form = deepcopy(self.TRANSACTIONS_FORM)
        else:
            _form = deepcopy(self.TRANSACTIONS_FORM)

        # make all hidden and readonly fields writable
        _form.set_all_readonly(False)

        for key, field_name in kw_map.items():
            if key in kwargs:
                # if the field is a date, format accordingly
                if key.endswith('_date'):
                    _form[field_name] = kwargs.get(key).strftime('%d/%m/%Y')
                else:
                    _form[field_name] = kwargs.get(key)

        try:
            r = self.post_url(self.TRANSACTIONS_EXPORT_URL, form=_form)
            return self._parse_transactions(r)
        except AuthRequiredException:
            self._auth()
            r = self.post_url(self.TRANSACTIONS_EXPORT_URL, form=_form)
            return self._parse_transactions(r)