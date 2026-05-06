def filter(self, query: Query, entity: type) -> Tuple[Query, Any]:
        """Apply the `_method` to all childs of the node.
        
        :param query: The sqlachemy query.
        :type query: Query

        :param entity: The entity model of the query.
        :type entity: type

        :return: A tuple with in first place the updated query and in second
            place the list of filters to apply to the query.
        :rtype: Tuple[Query, Any]
        """
        new_query = query
        c_filter_list = []
        for child in self._childs:
            new_query, f_list = child.filter(new_query, entity)
            c_filter_list.append(f_list)
        return (
            new_query,
            self._method(*c_filter_list)
        )