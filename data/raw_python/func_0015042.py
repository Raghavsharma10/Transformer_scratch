def create(self, session, values, *args, **kwargs):
        """
        Creates a new instance of the self.model
        and persists it to the database.

        :param dict values: The dictionary of values to
            set on the model.  The key is the column name
            and the value is what it will be set to.  If
            the cls._create_fields is defined then it will
            use those fields.  Otherwise, it will use the
            fields defined in cls.fields
        :param Session session: The sqlalchemy session
        :return: The serialized model.  It will use the self.fields
            attribute for this.
        :rtype: dict
        """
        model = self.model()
        model = self._set_values_on_model(model, values, fields=self.create_fields)
        session.add(model)
        session.commit()
        return self.serialize_model(model)