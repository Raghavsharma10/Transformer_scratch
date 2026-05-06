def comp(request, slug, directory_slug=None):
    """
    View the requested comp
    """
    context = {}
    path = settings.COMPS_DIR
    comp_dir = os.path.split(path)[1]
    template = "{0}/{1}".format(comp_dir, slug)
    if directory_slug:
        template = "{0}/{1}/{2}".format(comp_dir, directory_slug, slug)
    working_dir = os.path.join(path, slug)
    if os.path.isdir(working_dir):
        return redirect('comp-listing', directory_slug=slug)

    try:
        t = get_template(template)
    except TemplateDoesNotExist:
        return redirect('comp-listing')

    c = RequestContext(request, context)
    return HttpResponse(t.render(c))