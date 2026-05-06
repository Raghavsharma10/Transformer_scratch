def _only_trashed(self, builder):
        """
        The only-trashed extension.

        :param builder: The query builder
        :type builder: eloquent.orm.builder.Builder
        """
        model = builder.get_model()

        self.remove(builder, model)

        builder.get_query().where_not_null(model.get_qualified_deleted_at_column())