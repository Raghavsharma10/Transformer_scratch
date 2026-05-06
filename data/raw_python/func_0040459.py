def payment(self, amount, **kwargs):
        """Get payment URL and new transaction ID

        Usage::

            >>> import sofort
            >>> client = sofort.Client('123456', '123456', '123456',
                                       abort_url='https://mysite.com/abort')
            >>> t = client.pay(12, success_url='http://mysite.com?paid')

            >>> t.transaction
            123123-321231-56A3BE0E-ACAB
            >>> t.payment_url
            https://www.sofort.com/payment/go/136b2012718da216af4c20c2ec2f51100c90406e
        """
        params = self.config.clone()\
            .update({ 'amount': amount })\
            .update(kwargs)

        mandatory = ['abort_url', 'reasons', 'success_url']

        for field in mandatory:
            if not params.has(field):
                raise ValueError('Mandatory field "{}" is not specified'.format(field))

        params.reasons = [sofort.internals.strip_reason(reason)
                                for reason
                                in params.reasons]

        return self._request(sofort.xml.multipay(params), params)