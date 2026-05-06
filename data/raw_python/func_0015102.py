def do_direct_payment(self, paymentaction="Sale", **kwargs):
        """Shortcut for the DoDirectPayment method.

        ``paymentaction`` could be 'Authorization' or 'Sale'

        To issue a Sale immediately::

            charge = {
                'amt': '10.00',
                'creditcardtype': 'Visa',
                'acct': '4812177017895760',
                'expdate': '012010',
                'cvv2': '962',
                'firstname': 'John',
                'lastname': 'Doe',
                'street': '1 Main St',
                'city': 'San Jose',
                'state': 'CA',
                'zip': '95131',
                'countrycode': 'US',
                'currencycode': 'USD',
            }
            direct_payment("Sale", **charge)

        Or, since "Sale" is the default:

            direct_payment(**charge)

        To issue an Authorization, simply pass "Authorization" instead
        of "Sale".

        You may also explicitly set ``paymentaction`` as a keyword argument:

            ...
            direct_payment(paymentaction="Sale", **charge)
        """
        kwargs.update(self._sanitize_locals(locals()))
        return self._call('DoDirectPayment', **kwargs)