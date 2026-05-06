def transaction_search(self, **kwargs):
        """Shortcut for the TransactionSearch method.
        Returns a PayPalResponseList object, which merges the L_ syntax list
        to a list of dictionaries with properly named keys.

        Note that the API will limit returned transactions to 100.

        Required Kwargs
        ---------------
        * STARTDATE

        Optional Kwargs
        ---------------
        STATUS = one of ['Pending','Processing','Success','Denied','Reversed']

        """
        plain = self._call('TransactionSearch', **kwargs)
        return PayPalResponseList(plain.raw, self.config)