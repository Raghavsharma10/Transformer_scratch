def _merge_wheres_to_has(self, has_query, relation):
        """
        Merge the "wheres" from the relation query to a has query.

        :param has_query: The has query
        :type has_query: Builder

        :param relation: The relation to count
        :type relation: eloquent.orm.relations.Relation
        """
        relation_query = relation.get_base_query()

        has_query.merge_wheres(relation_query.wheres, relation_query.get_bindings())

        self._query.merge_bindings(has_query.get_query())