def taxids(self):
        """Distinct NCBI taxonomy identifiers (``taxid``) in :class:`.models.Entry`

        :return: NCBI taxonomy identifiers
        :rtype: list[int]
        """
        r = self.session.query(distinct(models.Entry.taxid)).all()
        return [x[0] for x in r]