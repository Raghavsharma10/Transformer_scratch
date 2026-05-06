def get_for_update(self, connection_name='DEFAULT', **kwargs):

        """
        http://docs.sqlalchemy.org/en/latest/orm/query.html?highlight=update#sqlalchemy.orm.query.Query.with_for_update  # noqa
        """
        if not kwargs:
            raise InvalidQueryError(
                "Can not execute a query without parameters")

        obj = self.pool.connections[connection_name].session.query(
            self._model).with_for_update(
                nowait=True, of=self._model).filter_by(**kwargs).first()
        if not obj:
            raise NotFoundError('Object not found')
        return obj