def dbreference_types(self):
        """Distinct database reference types (``type_``) in :class:`.models.DbReference`

        :return: List of strings for all available database cross reference types used in model DbReference
        :rtype: list[str]
        """
        q = self.session.query(distinct(models.DbReference.type_))
        return [x[0] for x in q.all()]