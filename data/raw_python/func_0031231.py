def tissues_in_references(self):
        """Distinct tissues (``tissue`` in :class:`.models.TissueInReference`)

        :return: all distinct tissues in references
        :rtype: list[str]
        """
        return [x[0] for x in self.session.query(models.TissueInReference.tissue).all()]