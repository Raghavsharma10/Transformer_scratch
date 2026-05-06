def page_revisions(request, page_id, template_name='wagtailrollbacks/edit_handlers/revisions.html'):
    """
    Returns GET response for specified page revisions.

    :param request: the request instance.
    :param page_id: the page ID.
    :param template_name: the template name.
    :rtype: django.http.HttpResponse.
    """
    page        = get_object_or_404(Page, pk=page_id)
    page_perms  = page.permissions_for_user(request.user)

    if not page_perms.can_edit():
        raise PermissionDenied

    page_num    = request.GET.get('p', 1)
    revisions   = get_revisions(page, page_num)

    return render(
        request,
        template_name,
        {
            'page':         page,
            'revisions':    revisions,
            'p':            page_num,
        }
    )