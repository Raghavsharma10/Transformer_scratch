def _join_tables(self, query: Query, join_models: Optional[List[type]]) -> Query:
        """Method to make the join when relation is found.

        :param query: The sqlalchemy query.
        :type query: Query

        :param join_models: The list of joined models get from the method
            `_get_relation`.
        :type join_models: Optional[List[type]]

        :return: The new Query with the joined tables.
        :rtype: Query
        """
        joined_query = query
        # Create the list of already joined entities
        joined_tables = [mapper.class_ for mapper in query._join_entities]
        if join_models:
            for j_model in join_models:
                if not j_model in joined_tables:
                    # /!\ join return a new query /!\
                    joined_query = joined_query.join(j_model)
        return joined_query