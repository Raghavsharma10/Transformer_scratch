def confirm_page_reversion(request, revision_id, template_name='wagtailrollbacks/pages/confirm_reversion.html'):
    """
    Handles page reversion process (GET and POST).

    :param request: the request instance.
    :param revision_id: the page revision ID.
    :param template_name: the template name.
    :rtype: django.http.HttpResponse.
    """
    revision    = get_object_or_404(PageRevision, pk=revision_id)
    page        = revision.page

    if page.locked:
        messages.error(
            request,
            _("Page '{0}' is locked.").format(page.title),
            buttons = []
        )

        return redirect(reverse('wagtailadmin_pages:edit', args=(page.id,)))

    page_perms = page.permissions_for_user(request.user)
    if not page_perms.can_edit():
        raise PermissionDenied

    if request.POST:
        is_publishing   = bool(request.POST.get('action-publish')) and page_perms.can_publish()
        is_submitting   = bool(request.POST.get('action-submit'))
        new_revision    = page.rollback(
            revision_id                 = revision_id,
            user                        = request.user,
            submitted_for_moderation    = is_submitting
        )

        if is_publishing:
            new_revision.publish()

            messages.success(
                request,
                _("Page '{0}' published.").format(page.title),
                buttons=[
                    messages.button(page.url, _('View live')),
                    messages.button(reverse('wagtailadmin_pages:edit', args=(page.id,)), _('Edit'))
                ]
            )
        elif is_submitting:
            messages.success(
                request,
                _("Page '{0}' submitted for moderation.").format(page.title),
                buttons=[
                    messages.button(reverse('wagtailadmin_pages:view_draft', args=(page.id,)), _('View draft')),
                    messages.button(reverse('wagtailadmin_pages:edit', args=(page.id,)), _('Edit'))
                ]
            )
            send_notification(new_revision.id, 'submitted', request.user.id)
        else:
            messages.success(
                request,
                _("Page '{0}' updated.").format(page.title),
                buttons=[]
            )

        for fn in hooks.get_hooks('after_edit_page'):
            result = fn(request, page)
            if hasattr(result, 'status_code'):
                return result

        return redirect('wagtailadmin_explore', page.get_parent().id)

    return render(
        request,
        template_name,
        {
            'page':         page,
            'revision':     revision,
            'page_perms':   page_perms
        }
    )