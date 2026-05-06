def revisions(
    request, slug, template_name='wakawaka/revisions.html', extra_context=None
):
    """
    Displays the list of all revisions for a specific WikiPage
    """
    queryset = WikiPage.objects.all()
    page = get_object_or_404(queryset, slug=slug)

    template_context = {'page': page}
    template_context.update(extra_context or {})
    return render(request, template_name, template_context)