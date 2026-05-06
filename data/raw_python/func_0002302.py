def get_content_item_inlines(plugins=None, base=BaseContentItemInline):
    """
    Dynamically generate genuine django inlines for all registered content item types.
    When the `plugins` parameter is ``None``, all plugin inlines are returned.
    """
    COPY_FIELDS = (
        'form', 'raw_id_fields', 'filter_vertical', 'filter_horizontal',
        'radio_fields', 'prepopulated_fields', 'formfield_overrides', 'readonly_fields',
    )
    if plugins is None:
        plugins = extensions.plugin_pool.get_plugins()

    inlines = []
    for plugin in plugins:  # self.model._supported_...()
        # Avoid errors that are hard to trace
        if not isinstance(plugin, extensions.ContentPlugin):
            raise TypeError("get_content_item_inlines() expects to receive ContentPlugin instances, not {0}".format(plugin))

        ContentItemType = plugin.model

        # Create a new Type that inherits CmsPageItemInline
        # Read the static fields of the ItemType to override default appearance.
        # This code is based on FeinCMS, (c) Simon Meers, BSD licensed
        class_name = '%s_AutoInline' %  ContentItemType.__name__
        attrs = {
            '__module__': plugin.__class__.__module__,
            'model': ContentItemType,

            # Add metadata properties for template
            'name': plugin.verbose_name,
            'plugin': plugin,
            'type_name': plugin.type_name,
            'extra_fieldsets': plugin.fieldsets,
            'cp_admin_form_template': plugin.admin_form_template,
            'cp_admin_init_template': plugin.admin_init_template,
        }

        # Copy a restricted set of admin fields to the inline model too.
        for name in COPY_FIELDS:
            if getattr(plugin, name):
                attrs[name] = getattr(plugin, name)

        inlines.append(type(class_name, (base,), attrs))

    # For consistency, enforce ordering
    inlines.sort(key=lambda inline: inline.name.lower())

    return inlines