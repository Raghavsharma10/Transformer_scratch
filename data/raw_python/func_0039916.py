def _render_email_placeholder(request, email_template, base_url, context):
    """
    Internal rendering of the placeholder/contentitems.

    This a simple variation of render_placeholder(),
    making is possible to render both a HTML and text item in a single call.
    Caching is currently not implemented.

    :rtype: fluentcms_emailtemplates.rendering.EmailBodyContent
    """
    placeholder = email_template.contents
    items = placeholder.get_content_items(email_template)

    if not items:  # NOTES: performs query
        # There are no items, fetch the fallback language.
        language_code = fc_appsettings.FLUENT_CONTENTS_DEFAULT_LANGUAGE_CODE
        items = placeholder.get_content_items(email_template, limit_parent_language=False).translated(language_code)

    html_fragments = []
    text_fragments = []

    for instance in items:
        plugin = instance.plugin
        html_part = _render_html(plugin, request, instance, context)
        text_part = _render_text(plugin, request, instance, context, base_url)
        html_fragments.append(html_part)
        text_fragments.append(text_part)

    html_body = u"".join(html_fragments)
    text_body = u"".join(text_fragments)

    return EmailBodyContent(text_body, html_body)