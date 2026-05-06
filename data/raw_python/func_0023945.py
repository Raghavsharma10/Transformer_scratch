def debit(self, amount, credit_account, description, debit_memo="", credit_memo="", datetime=None):
        """ Post a debit of 'amount' and a credit of -amount against this account and credit_account respectively.

        note amount must be non-negative.
        """

        assert amount >= 0
        return self.post(amount, credit_account, description, self_memo=debit_memo, other_memo=credit_memo, datetime=datetime)