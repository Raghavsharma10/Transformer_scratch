def feature_types(self):
        """Distinct types (``type_``) in :class:`.models.Feature`

        :return: all distinct feature types
        :rtype: list[str]
        """
        r = self.session.query(distinct(models.Feature.type_)).all()
        return [x[0] for x in r]