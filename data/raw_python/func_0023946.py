def credit(self, amount, debit_account, description, debit_memo="", credit_memo="", datetime=None):
        """ Post a credit of 'amount' and a debit of -amount against this account and credit_account respectively.

        note amount must be non-negative.
        """
        assert amount >= 0
        return self.post(-amount, debit_account, description, self_memo=credit_memo, other_memo=debit_memo, datetime=datetime)