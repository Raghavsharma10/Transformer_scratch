def render_content_items(request, items, template_name=None, cachable=None):
    """
    Render a list of :class:`~fluent_contents.models.ContentItem` objects as HTML string.
    This is a variation of the :func:`render_placeholder` function.

    Note that the items are not filtered in any way by parent or language.
    The items are rendered as-is.

    :param request: The current request object.
    :type request: :class:`~django.http.HttpRequest`
    :param items: The list or queryset of objects to render. Passing a queryset is preferred.
    :type items: list or queryset of :class:`~fluent_contents.models.ContentItem`.
    :param template_name: Optional template name used to concatenate the placeholder output.
    :type template_name: Optional[str]
    :param cachable: Whether the output is cachable, otherwise the full output will not be cached.
                     Default: False when using a template, True otherwise.
    :type cachable: Optional[bool]

    :rtype: :class:`~fluent_contents.models.ContentItemOutput`
    """
    if not items:
        output = ContentItemOutput(mark_safe(u"<!-- no items to render -->"))
    else:
        output = RenderingPipe(request).render_items(
            placeholder=None,
            items=items,
            parent_object=None,
            template_name=template_name,
            cachable=cachable
        )

    # Wrap the result after it's stored in the cache.
    if markers.is_edit_mode(request):
        output.html = markers.wrap_anonymous_output(output.html)

    return output