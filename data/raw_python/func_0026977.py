def _preprocess_kwargs(self, initial_kwargs):
        """ Replace generic key related attribute with filters by object_id and content_type fields """
        kwargs = initial_kwargs.copy()
        generic_key_related_kwargs = self._get_generic_key_related_kwargs(initial_kwargs)
        for key, value in generic_key_related_kwargs.items():
            # delete old kwarg that was related to generic key
            del kwargs[key]
            try:
                suffix = key.split('__')[1]
            except IndexError:
                suffix = None
            # add new kwargs that related to object_id and content_type fields
            new_kwargs = self._get_filter_object_id_and_content_type_filter_kwargs(value, suffix)
            kwargs.update(new_kwargs)

        return kwargs