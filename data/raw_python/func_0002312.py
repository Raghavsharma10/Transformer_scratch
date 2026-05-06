def get_search_field_values(contentitem):
    """
    Extract the search fields from the model.
    """
    plugin = contentitem.plugin
    values = []
    for field_name in plugin.search_fields:
        value = getattr(contentitem, field_name)

        # Just assume all strings may contain HTML.
        # Not checking for just the PluginHtmlField here.
        if value and isinstance(value, six.string_types):
            value = get_cleaned_string(value)

        values.append(value)

    return values