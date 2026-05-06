def make_payment(self, recipient, amount, description=None):
        """
        make_payment allows for automated payments.
        A use case includes the ability to trigger a payment to a customer
        who requires a refund for example.
        You only need to provide the recipient and the amount to be transfered.
        It supports transfers in Kobo as well, just add the necessary decimals.
        """
        self.br.open(self.MOBILE_WEB_URL % {'accountno': self.account})
        try:
            # Search for the existence of the Register link - indicating a new account
            # Payments cannot be made from an account that isn't registered - obviously.
            self.br.find_link(text='Register')
            raise InvalidAccountException
        except mechanize.LinkNotFoundError:
            pass

        # Follow through by clicking links to get to the payment form
        self.br.follow_link(text='My sarafu')
        self.br.follow_link(text='Transfer')
        self.br.follow_link(text='sarafu-to-sarafu')
        self.br.select_form(nr=0)
        self.br['recipient'] = recipient
        self.br.new_control('text', 'pin', attrs={'value': self.pin})
        self.br.new_control('text', 'amount', attrs={'value': amount})
        self.br.new_control('text', 'channel', attrs={'value': 'WAP'})
        r = self.br.submit()

        # Right away, we can tell if the recipient doesn't exist and raise an exception
        if re.search(r'Recipient not found', r):
            raise InvalidAccountException

        self.br.select_form(nr=0)
        self.br.new_control('text', 'pin', attrs={'value': self.pin})
        r = self.br.submit().read()

        # We don't get to know if our pin was valid until this step
        if re.search(r'Invalid PIN', r):
            raise AuthDeniedException

        # An error could occur for other reasons
        if re.search(r'Error occured', r):
            raise RequestErrorException

        # If it was successful, we extract the transaction id and return that
        if re.search(r'Transaction id: (?P<txnid>\d+)', r):
            match = re.search(r'Transaction id: (?P<txnid>\d+)', r)
            return match.group('txnid')