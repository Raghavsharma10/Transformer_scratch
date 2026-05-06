def ledger(self, start=None, end=None):
        """Returns a list of entries for this account.

        Ledger returns a sequence of LedgerEntry's matching the criteria
        in chronological order. The returned sequence can be boolean-tested
        (ie. test that nothing was returned).

        If 'start' is given, only entries on or after that datetime are
        returned.  'start' must be given with a timezone.

        If 'end' is given, only entries before that datetime are
        returned.  'end' must be given with a timezone.
        """

        DEBIT_IN_DB = self._DEBIT_IN_DB()

        flip = 1
        if self._positive_credit():
            flip *= -1

        qs = self._entries_range(start=start, end=end)
        qs = qs.order_by("transaction__t_stamp", "transaction__tid")

        balance = Decimal("0.00")
        if start:
            balance = self.balance(start)

        if not qs:
            return []

        #helper is a hack so the caller can test for no entries.
        def helper(balance_in):
            balance = balance_in
            for e in qs.all():
                amount = e.amount * DEBIT_IN_DB
                o_balance = balance
                balance += flip * amount

                yield LedgerEntry(amount, e, o_balance, balance)

        return helper(balance)