def do_capture(self, authorizationid, amt, completetype='Complete',
                   **kwargs):
        """Shortcut for the DoCapture method.

        Use the TRANSACTIONID from DoAuthorization, DoDirectPayment or
        DoExpressCheckoutPayment for the ``authorizationid``.

        The `amt` should be the same as the authorized transaction.
        """
        kwargs.update(self._sanitize_locals(locals()))
        return self._call('DoCapture', **kwargs)