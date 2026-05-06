def confirm(self, token=None):
        """Returns the status of the invoice

        STATUSES: pending, completed, cancelled
        """
        _token = token if token else self._response.get("token")
        return self._process('checkout-invoice/confirm/' + str(_token))