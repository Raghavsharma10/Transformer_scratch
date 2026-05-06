def filter(self, query: Query, entity: type) -> Tuple[Query, Any]:
        """Define the filter function that every node must to implement.

        :param query: The sqlalchemy query.
        :type query: Query

        :param entity: The entity model.
        :type entity: type

        :return: The filtered query.
        :rtype: Tuple[Query, Any]
        """
        raise NotImplementedError('You must implement this.')