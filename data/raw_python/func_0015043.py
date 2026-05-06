def retrieve(self, session, lookup_keys, *args, **kwargs):
        """
        Retrieves a model using the lookup keys provided.
        Only one model should be returned by the lookup_keys
        or else the manager will fail.

        :param Session session: The SQLAlchemy session to use
        :param dict lookup_keys: A dictionary mapping the fields
            and their expected values
        :return: The dictionary of keys and values for the retrieved
            model.  The only values returned will be those specified by
            fields attrbute on the class
        :rtype: dict
        :raises: NotFoundException
        """
        model = self._get_model(lookup_keys, session)
        return self.serialize_model(model)