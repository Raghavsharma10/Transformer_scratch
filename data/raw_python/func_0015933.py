def page_list(
    request, template_name='wakawaka/page_list.html', extra_context=None
):
    """
    Displays all Pages
    """
    page_list = WikiPage.objects.all()
    page_list = page_list.order_by('slug')

    template_context = {
        'page_list': page_list,
        'index_slug': getattr(settings, 'WAKAWAKA_DEFAULT_INDEX', 'WikiIndex'),
    }
    template_context.update(extra_context or {})
    return render(request, template_name, template_context)