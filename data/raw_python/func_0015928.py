def page(
    request,
    slug,
    rev_id=None,
    template_name='wakawaka/page.html',
    extra_context=None,
):
    """
    Displays a wiki page. Redirects to the edit view if the page doesn't exist.
    """
    try:
        queryset = WikiPage.objects.all()
        page = queryset.get(slug=slug)
        rev = page.current

        # Display an older revision if rev_id is given
        if rev_id:
            revision_queryset = Revision.objects.all()
            rev_specific = revision_queryset.get(pk=rev_id)
            if rev.pk != rev_specific.pk:
                rev_specific.is_not_current = True
            rev = rev_specific

    # The Page does not exist, redirect to the edit form or
    # deny, if the user has no permission to add pages
    except WikiPage.DoesNotExist:
        if request.user.is_authenticated:
            kwargs = {'slug': slug}
            redirect_to = reverse('wakawaka_edit', kwargs=kwargs)
            return HttpResponseRedirect(redirect_to)
        raise Http404
    template_context = {'page': page, 'rev': rev}
    template_context.update(extra_context or {})
    return render(request, template_name, template_context)