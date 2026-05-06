def refund_transaction(self, transactionid=None, payerid=None, **kwargs):
        """Shortcut for RefundTransaction method.
           Note new API supports passing a PayerID instead of a transaction id,
           exactly one must be provided.

           Optional:
               INVOICEID
               REFUNDTYPE
               AMT
               CURRENCYCODE
               NOTE
               RETRYUNTIL
               REFUNDSOURCE
               MERCHANTSTOREDETAILS
               REFUNDADVICE
               REFUNDITEMDETAILS
               MSGSUBID

           MERCHANSTOREDETAILS has two fields:
               STOREID
               TERMINALID
           """
        # This line seems like a complete waste of time... kwargs should not
        # be populated
        if (transactionid is None) and (payerid is None):
            raise PayPalError(
                'RefundTransaction requires either a transactionid or '
                'a payerid')
        if (transactionid is not None) and (payerid is not None):
            raise PayPalError(
                'RefundTransaction requires only one of transactionid %s '
                'and payerid %s' % (transactionid, payerid))
        if transactionid is not None:
            kwargs['TRANSACTIONID'] = transactionid
        else:
            kwargs['PAYERID'] = payerid

        return self._call('RefundTransaction', **kwargs)