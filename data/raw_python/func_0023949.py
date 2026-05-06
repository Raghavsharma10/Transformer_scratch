def totals(self, start=None, end=None):
        """Returns a Totals object containing the sum of all debits, credits
        and net change over the period of time from start to end.

        'start' is inclusive, 'end' is exclusive
        """

        qs = self._entries_range(start=start, end=end)
        qs_positive = qs.filter(amount__gt=Decimal("0.00")).all().aggregate(Sum('amount'))
        qs_negative = qs.filter(amount__lt=Decimal("0.00")).all().aggregate(Sum('amount'))

        #Is there a cleaner way of saying this?  Should the sum of 0 things be None?
        positives = qs_positive['amount__sum'] if qs_positive['amount__sum'] is not None else 0
        negatives = -qs_negative['amount__sum'] if qs_negative['amount__sum'] is not None else 0

        if self._DEBIT_IN_DB() > 0:
            debits = positives
            credits = negatives
        else:
            debits = negatives
            credits = positives

        net = debits-credits
        if self._positive_credit():
            net = -net

        return self.Totals(credits, debits, net)