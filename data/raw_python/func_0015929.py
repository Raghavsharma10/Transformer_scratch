def edit(
    request,
    slug,
    rev_id=None,
    template_name='wakawaka/edit.html',
    extra_context=None,
    wiki_page_form=WikiPageForm,
    wiki_delete_form=DeleteWikiPageForm,
):
    """
    Displays the form for editing and deleting a page.
    """
    # Get the page for slug and get a specific revision, if given
    try:
        queryset = WikiPage.objects.all()
        page = queryset.get(slug=slug)
        rev = page.current
        initial = {'content': page.current.content}

        # Do not allow editing wiki pages if the user has no permission
        if not request.user.has_perms(
            ('wakawaka.change_wikipage', 'wakawaka.change_revision')
        ):
            return HttpResponseForbidden(
                ugettext('You don\'t have permission to edit pages.')
            )

        if rev_id:
            # There is a specific revision, fetch this
            rev_specific = Revision.objects.get(pk=rev_id)
            if rev.pk != rev_specific.pk:
                rev = rev_specific
                rev.is_not_current = True
                initial = {
                    'content': rev.content,
                    'message': _('Reverted to "%s"' % rev.message),
                }

    # This page does not exist, create a dummy page
    # Note that it's not saved here
    except WikiPage.DoesNotExist:

        # Do not allow adding wiki pages if the user has no permission
        if not request.user.has_perms(
            ('wakawaka.add_wikipage', 'wakawaka.add_revision')
        ):
            return HttpResponseForbidden(
                ugettext('You don\'t have permission to add wiki pages.')
            )

        page = WikiPage(slug=slug)
        page.is_initial = True
        rev = None
        initial = {
            'content': _('Describe your new page %s here...' % slug),
            'message': _('Initial revision'),
        }

    # Don't display the delete form if the user has nor permission
    delete_form = None
    # The user has permission, then do
    if request.user.has_perm(
        'wakawaka.delete_wikipage'
    ) or request.user.has_perm('wakawaka.delete_revision'):
        delete_form = wiki_delete_form(request)
        if request.method == 'POST' and request.POST.get('delete'):
            delete_form = wiki_delete_form(request, request.POST)
            if delete_form.is_valid():
                return delete_form.delete_wiki(request, page, rev)

    # Page add/edit form
    form = wiki_page_form(initial=initial)
    if request.method == 'POST':
        form = wiki_page_form(data=request.POST)
        if form.is_valid():
            # Check if the content is changed, except there is a rev_id and the
            # user possibly only reverted the HEAD to it
            if (
                not rev_id
                and initial['content'] == form.cleaned_data['content']
            ):
                form.errors['content'] = (_('You have made no changes!'),)

            # Save the form and redirect to the page view
            else:
                try:
                    # Check that the page already exist
                    queryset = WikiPage.objects.all()
                    page = queryset.get(slug=slug)
                except WikiPage.DoesNotExist:
                    # Must be a new one, create that page
                    page = WikiPage(slug=slug)
                    page.save()

                form.save(request, page)

                kwargs = {'slug': page.slug}

                redirect_to = reverse('wakawaka_page', kwargs=kwargs)
                messages.success(
                    request,
                    ugettext('Your changes to %s were saved' % page.slug),
                )
                return HttpResponseRedirect(redirect_to)

    template_context = {
        'form': form,
        'delete_form': delete_form,
        'page': page,
        'rev': rev,
    }
    template_context.update(extra_context or {})
    return render(request, template_name, template_context)