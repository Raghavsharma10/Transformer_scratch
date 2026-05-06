def get_options_info(self, symbol, items=None, expiration=''):
        """get_options_data() uses the yahoo.finance.options table to retrieve call and put options from the options page.
        """
        response = self.select('yahoo.finance.options',items).where(['symbol','=',symbol],[] if not expiration else ['expiration','=',expiration])
        return response