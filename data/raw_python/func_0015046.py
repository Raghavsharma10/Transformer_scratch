def delete(self, session, lookup_keys, *args, **kwargs):
        """
        Deletes the model found using the lookup_keys

        :param Session session: The SQLAlchemy session to use
        :param dict lookup_keys: A dictionary mapping the fields
            and their expected values
        :return: An empty dictionary
        :rtype: dict
        :raises: NotFoundException
        """
        model = self._get_model(lookup_keys, session)
        session.delete(model)
        session.commit()
        return {}