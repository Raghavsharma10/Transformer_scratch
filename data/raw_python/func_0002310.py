def get_render_language(contentitem):
    """
    Tell which language should be used to render the content item.
    """
    plugin = contentitem.plugin

    if plugin.render_ignore_item_language \
    or (plugin.cache_output and plugin.cache_output_per_language):
        # Render the template in the current language.
        # The cache also stores the output under the current language code.
        #
        # It would make sense to apply this for fallback content too,
        # but that would be ambiguous however because the parent_object could also be a fallback,
        # and that case can't be detected here. Hence, better be explicit when desiring multi-lingual content.
        return get_language()  # Avoid switching the content,
    else:
        # Render the template in the ContentItem language.
        # This makes sure that {% trans %} tag output matches the language of the model field data.
        return contentitem.language_code