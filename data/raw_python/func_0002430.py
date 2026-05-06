def remove(self, builder, model):
        """
        Remove the scope from a given query builder.

        :param builder: The query builder
        :type builder: eloquent.orm.builder.Builder

        :param model: The model
        :type model: eloquent.orm.Model
        """
        column = model.get_qualified_deleted_at_column()

        query = builder.get_query()

        wheres = []
        for where in query.wheres:
            # If the where clause is a soft delete date constraint,
            # we will remove it from the query and reset the keys
            # on the wheres. This allows the developer to include
            # deleted model in a relationship result set that is lazy loaded.
            if not self._is_soft_delete_constraint(where, column):
                wheres.append(where)

        query.wheres = wheres