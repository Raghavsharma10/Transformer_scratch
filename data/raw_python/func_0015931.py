def changes(
    request, slug, template_name='wakawaka/changes.html', extra_context=None
):
    """
    Displays the changes between two revisions.
    """
    rev_a_id = request.GET.get('a', None)
    rev_b_id = request.GET.get('b', None)

    # Some stinky fingers manipulated the url
    if not rev_a_id or not rev_b_id:
        return HttpResponseBadRequest('Bad Request')

    try:
        revision_queryset = Revision.objects.all()
        wikipage_queryset = WikiPage.objects.all()
        rev_a = revision_queryset.get(pk=rev_a_id)
        rev_b = revision_queryset.get(pk=rev_b_id)
        page = wikipage_queryset.get(slug=slug)
    except ObjectDoesNotExist:
        raise Http404

    if rev_a.content != rev_b.content:
        d = difflib.unified_diff(
            rev_b.content.splitlines(),
            rev_a.content.splitlines(),
            'Original',
            'Current',
            lineterm='',
        )
        difftext = '\n'.join(d)
    else:
        difftext = _(u'No changes were made between this two files.')

    template_context = {
        'page': page,
        'diff': difftext,
        'rev_a': rev_a,
        'rev_b': rev_b,
    }
    template_context.update(extra_context or {})
    return render(request, template_name, template_context)