def create_for_placeholder(self, placeholder, sort_order=1, language_code=None, **kwargs):
        """
        Create a Content Item with the given parameters

        If the language_code is not provided, the language code of the parent will be used.
        This may perform an additional database query, unless
        the :class:`~fluent_contents.models.managers.PlaceholderManager` methods were used to construct the object,
        such as :func:`~fluent_contents.models.managers.PlaceholderManager.create_for_object`
        or :func:`~fluent_contents.models.managers.PlaceholderManager.get_by_slot`
        """
        if language_code is None:
            # Could also use get_language() or appsettings.FLUENT_CONTENTS_DEFAULT_LANGUAGE_CODE
            # thus avoid the risk of performing an extra query here to the parent.
            # However, this identical behavior to BaseContentItemFormSet,
            # and the parent can be set already via Placeholder.objects.create_for_object()
            language_code = get_parent_language_code(placeholder.parent)

        obj = self.create(
            placeholder=placeholder,
            parent_type_id=placeholder.parent_type_id,
            parent_id=placeholder.parent_id,
            sort_order=sort_order,
            language_code=language_code,
            **kwargs
        )

        # Fill the reverse caches
        obj.placeholder = placeholder
        parent = getattr(placeholder, '_parent_cache', None)  # by GenericForeignKey (_meta.virtual_fields[0].cache_attr)
        if parent is not None:
            obj.parent = parent

        return obj