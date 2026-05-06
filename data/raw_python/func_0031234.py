def version(self):
        """Version of UniPort knowledgebase

        :returns: dictionary with version info
        :rtype: dict
        """
        return [x for x in self.session.query(models.Version).all()]