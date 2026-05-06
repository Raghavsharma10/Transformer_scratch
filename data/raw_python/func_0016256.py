def clean_update_fields(self, index, update_fields):
        """
        Clean the list of update_fields based on the index being updated.\

        If any field in the update_fields list is not in the set of properties
        defined by the index mapping for this model, then we ignore it. If
        a field _is_ in the mapping, but the underlying model field is a
        related object, and thereby not directly serializable, then this
        method will raise a ValueError.

        """
        search_fields = get_model_index_properties(self, index)
        clean_fields = [f for f in update_fields if f in search_fields]
        ignore = [f for f in update_fields if f not in search_fields]
        if ignore:
            logger.debug(
                "Ignoring fields from partial update: %s",
                [f for f in update_fields if f not in search_fields],
            )
        for f in clean_fields:
            if not self._is_field_serializable(f):
                raise ValueError(
                    "'%s' cannot be automatically serialized into a search document property. Please override as_search_document_update.",
                    f,
                )
        return clean_fields