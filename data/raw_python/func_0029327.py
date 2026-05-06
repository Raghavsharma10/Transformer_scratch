def set_currency(self, currency_id):
        """
        Set transaction currency code from given currency id, e.g. set 840 from 'USD'
        """
        try:
            self.currency = currency_codes[currency_id]
            self.IsoMessage.FieldData(49, self.currency)
            self.rebuild()
        except KeyError:
            self.currency = None