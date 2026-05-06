def report(request, comment_id):
    """
    Flags a comment on GET.

    Redirects to whatever is provided in request.REQUEST['next'].
    """

    comment = get_object_or_404(
        django_comments.get_model(), pk=comment_id, site__pk=settings.SITE_ID)
    if comment.parent is not None:
        messages.info(request, _('Reporting comment replies is not allowed.'))
    else:
        perform_flag(request, comment)
        messages.info(request, _('The comment has been reported.'))

    next = request.GET.get('next') or comment.get_absolute_url()
    return HttpResponseRedirect(next)