def _get_model_fields(self, model, prefix=_field_prefix):
        """
        Find all fields of given model that are not default models.
        """
        fields = list()
        for field_name, field in model()._ordered_fields:
            # Filter the default fields
            if field_name not in getattr(model, '_DEFAULT_BASE_FIELDS', []):
                type_name = utils.to_camel(field.solr_type)
                required = self._marker_true if field.required is True else self._marker_false
                fields.append((prefix, field_name, type_name, required))

        return fields