def _get_model(self, lookup_keys, session):
        """
        Gets the sqlalchemy Model instance associated with
        the lookup keys.

        :param dict lookup_keys: A dictionary of the keys
            and their associated values.
        :param Session session: The sqlalchemy session
        :return: The sqlalchemy orm model instance.
        """
        try:
            return self.queryset(session).filter_by(**lookup_keys).one()
        except NoResultFound:
            raise NotFoundException('No model of type {0} was found using '
                                    'lookup_keys {1}'.format(self.model.__name__, lookup_keys))