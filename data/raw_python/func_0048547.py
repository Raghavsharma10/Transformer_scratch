def loan(self):
        """
        str: representation of the ``LOAN`` spoolverb. E.g.:
            ``'ASCRIBESPOOL01LOAN1/150526150528'``.
        """
        return '{}{}LOAN{}/{}{}'.format(self.meta, self.version, self.edition_number,
                                        self.loan_start, self.loan_end)