def preview_page_version(request, revision_id):
    """
    Returns GET response for specified page preview.

    :param request: the request instance.
    :param reversion_pk: the page revision ID.
    :rtype: django.http.HttpResponse.
    """
    revision = get_object_or_404(PageRevision, pk=revision_id)

    if not revision.page.permissions_for_user(request.user).can_publish():
        raise PermissionDenied

    page                = revision.as_page_object()
    request.revision_id = revision_id

    return page.serve_preview(request, page.default_preview_mode)