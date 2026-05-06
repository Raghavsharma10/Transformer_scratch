def extend(self, builder):
        """
        Extend the query builder with the needed functions.

        :param builder: The query builder
        :type builder: eloquent.orm.builder.Builder
        """
        for extension in self._extensions:
            getattr(self, '_add_%s' % extension)(builder)

        builder.on_delete(self._on_delete)