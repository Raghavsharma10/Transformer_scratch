def token(self):
        """
        Token given by Transbank for payment initialization url.

        Will raise PaymentError when an error ocurred.
        """
        if not self._token:
            self._token = self.fetch_token()
            logger.payment(self)
        return self._token