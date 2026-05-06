def update(self, session, lookup_keys, updates, *args, **kwargs):
        """
        Updates the model with the specified lookup_keys and returns
        the dictified object.

        :param Session session: The SQLAlchemy session to use
        :param dict lookup_keys: A dictionary mapping the fields
            and their expected values
        :param dict updates: The columns and the values to update
            them to.
        :return: The dictionary of keys and values for the retrieved
            model.  The only values returned will be those specified by
            fields attrbute on the class
        :rtype: dict
        :raises: NotFoundException
        """
        model = self._get_model(lookup_keys, session)
        model = self._set_values_on_model(model, updates, fields=self.update_fields)
        session.commit()
        return self.serialize_model(model)