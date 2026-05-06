def get_balance(self):
        """
        Retrieves the balance for the configured account
        """
        self.br.open(self.MOBILE_WEB_URL % {'accountno': self.account})
        try:
            # Search for the existence of the Register link - indicating a new account
            self.br.find_link(text='Register')
            raise InvalidAccountException
        except mechanize.LinkNotFoundError:
            pass

        self.br.follow_link(text='My sarafu')
        self.br.follow_link(text='Balance Inquiry')
        self.br.select_form(nr=0)
        self.br['pin'] = self.pin
        r = self.br.submit().read()

        # Pin valid?
        if re.search(r'Invalid PIN', r):
            raise AuthDeniedException

        # An error could occur for other reasons
        if re.search(r'Error occured', r):
            raise RequestErrorException

        # If it was successful, we extract the balance
        if re.search(r'Your balance is TSH (?P<balance>[\d\.]+)', r):
            match = re.search(r'Your balance is TSH (?P<balance>[\d\.]+)', r)
            return match.group('balance')