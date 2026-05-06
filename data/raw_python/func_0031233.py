def diseases(self):
        """Distinct diseases (``name`` in :class:`.models.Disease`)

        :returns: all distinct disease names
        :rtype: list[str]
        """
        return [x[0] for x in self.session.query(models.Disease.name).all()]