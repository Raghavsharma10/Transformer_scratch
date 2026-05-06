def revision_list(
    request, template_name='wakawaka/revision_list.html', extra_context=None
):
    """
    Displays a list of all recent revisions.
    """
    revision_list = Revision.objects.all()
    template_context = {'revision_list': revision_list}
    template_context.update(extra_context or {})
    return render(request, template_name, template_context)