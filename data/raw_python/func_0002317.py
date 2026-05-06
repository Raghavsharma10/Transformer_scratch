def render_placeholder(request, placeholder, parent_object=None, template_name=None, cachable=None, limit_parent_language=True, fallback_language=None):
    """
    Render a :class:`~fluent_contents.models.Placeholder` object.
    Returns a :class:`~fluent_contents.models.ContentItemOutput` object
    which contains the HTML output and :class:`~django.forms.Media` object.

    This function also caches the complete output of the placeholder
    when all individual items are cacheable.

    :param request: The current request object.
    :type request: :class:`~django.http.HttpRequest`
    :param placeholder: The placeholder object.
    :type placeholder: :class:`~fluent_contents.models.Placeholder`
    :param parent_object: Optional, the parent object of the placeholder (already implied by the placeholder)
    :param template_name: Optional template name used to concatenate the placeholder output.
    :type template_name: str | None
    :param cachable: Whether the output is cachable, otherwise the full output will not be cached.
                     Default: False when using a template, True otherwise.
    :type cachable: bool | None
    :param limit_parent_language: Whether the items should be limited to the parent language.
    :type limit_parent_language: bool
    :param fallback_language: The fallback language to use if there are no items in the current language. Passing ``True`` uses the default :ref:`FLUENT_CONTENTS_DEFAULT_LANGUAGE_CODE`.
    :type fallback_language: bool/str
    :rtype: :class:`~fluent_contents.models.ContentItemOutput`
    """
    output = PlaceholderRenderingPipe(request).render_placeholder(
        placeholder=placeholder,
        parent_object=parent_object,
        template_name=template_name,
        cachable=cachable,
        limit_parent_language=limit_parent_language,
        fallback_language=fallback_language
    )

    # Wrap the result after it's stored in the cache.
    if markers.is_edit_mode(request):
        output.html = markers.wrap_placeholder_output(output.html, placeholder)

    return output