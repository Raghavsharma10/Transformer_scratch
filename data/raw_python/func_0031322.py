def clean_existing(self, value):
        """Clean the data and return an existing document with its fields
        updated based on the cleaned values.
        """
        existing_pk = value[self.pk_field]
        try:
            obj = self.fetch_existing(existing_pk)
        except ReferenceNotFoundError:
            raise ValidationError('Object does not exist.')
        orig_data = self.get_orig_data_from_existing(obj)

        # Clean the data (passing the new data dict and the original data to
        # the schema).
        value = self.schema_class(value, orig_data).full_clean()

        # Set cleaned data on the object (except for the pk_field).
        for field_name, field_value in value.items():
            if field_name != self.pk_field:
                setattr(obj, field_name, field_value)

        return obj