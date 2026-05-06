def datasets(self):
        """Distinct datasets (``dataset``) in :class:`.models.Entry`

        Distinct datasets are SwissProt or/and TrEMBL

        :return: all distinct dataset types
        :rtype: list[str]
        """
        r = self.session.query(distinct(models.Entry.dataset)).all()
        return [x[0] for x in r]