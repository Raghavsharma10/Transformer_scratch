def _get_has_relation_query(self, relation):
        """
        Get the "has" relation base query

        :type relation: str

        :rtype: Builder
        """
        from .relations import Relation

        return Relation.no_constraints(
            lambda: getattr(self.get_model(), relation)()
        )