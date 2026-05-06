def post(self, amount, other_account, description, self_memo="", other_memo="", datetime=None):
        """ Post a transaction of 'amount' against this account and the negative amount against 'other_account'.

        This will show as a debit or credit against this account when amount > 0 or amount < 0 respectively.
        """

        #Note: debits are always positive, credits are always negative.  They should be negated before displaying
        #(expense and liability?) accounts
        tx = self._new_transaction()

        if datetime:
            tx.t_stamp = datetime
        #else now()

        tx.description = description
        tx.save()

        a1 = self._make_ae(self._DEBIT_IN_DB() * amount, self_memo, tx)
        a1.save()
        a2 = other_account._make_ae(-self._DEBIT_IN_DB() * amount, other_memo, tx)
        a2.save()

        return (a1, a2)