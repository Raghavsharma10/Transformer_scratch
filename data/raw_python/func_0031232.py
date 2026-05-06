def keywords(self):
        """Distinct keywords (``name`` in :class:`.models.Keyword`)

        :returns: all distinct keywords
        :rtype: list[str]
        """
        return [x[0] for x in self.session.query(models.Keyword.name).all()]