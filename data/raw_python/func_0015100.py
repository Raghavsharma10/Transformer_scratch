def do_authorization(self, transactionid, amt):
        """Shortcut for the DoAuthorization method.

        Use the TRANSACTIONID from DoExpressCheckoutPayment for the
        ``transactionid``. The latest version of the API does not support the
        creation of an Order from `DoDirectPayment`.

        The `amt` should be the same as passed to `DoExpressCheckoutPayment`.

        Flow for a payment involving a `DoAuthorization` call::

             1. One or many calls to `SetExpressCheckout` with pertinent order
                details, returns `TOKEN`
             1. `DoExpressCheckoutPayment` with `TOKEN`, `PAYMENTACTION` set to
                Order, `AMT` set to the amount of the transaction, returns
                `TRANSACTIONID`
             1. `DoAuthorization` with `TRANSACTIONID` and `AMT` set to the
                amount of the transaction.
             1. `DoCapture` with the `AUTHORIZATIONID` (the `TRANSACTIONID`
                returned by `DoAuthorization`)

        """
        args = self._sanitize_locals(locals())
        return self._call('DoAuthorization', **args)