def balance(self, date=None):
        """ returns the account balance as of 'date' (datetime stamp) or now().  """

        qs = self._entries()
        if date:
            qs = qs.filter(transaction__t_stamp__lt=date)
        r = qs.aggregate(b=Sum('amount'))
        b = r['b']

        flip = self._DEBIT_IN_DB()
        if self._positive_credit():
            flip *= -1

        if b is None:
            b = Decimal("0.00")
        b *= flip

        #print "returning balance %s for %s" % (b, self)
        return b