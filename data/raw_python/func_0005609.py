def filter(self, query: Query):
        """Return a new filtered query.
        
        Use the tree to filter the query and return a new query "filtered".
        This query can be filtered again using another tree or even a manual
        filter.
        To manually filter query see :
         - https://docs.sqlalchemy.org/en/rel_1_2/orm/query.html?highlight=filter#sqlalchemy.orm.query.Query.filter
        """
        entity = query.column_descriptions[0]['type']
        new_query, filters = self._root.filter(query, entity)
        return new_query.filter(filters)