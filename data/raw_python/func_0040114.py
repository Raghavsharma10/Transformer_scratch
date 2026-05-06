def comp_listing(request, directory_slug=None):
    """
    Output the list of HTML templates and subdirectories in the COMPS_DIR
    """
    context = {}
    working_dir = settings.COMPS_DIR
    if directory_slug:
        working_dir = os.path.join(working_dir, directory_slug)
    dirnames = []
    templates = []
    items = os.listdir(working_dir)
    templates = [x for x in items if os.path.splitext(x)[1] == '.html']
    dirnames = [x for x in items if \
                    not os.path.isfile(os.path.join(working_dir, x))]
    templates.sort()
    dirnames.sort()
    context['directories'] = dirnames
    context['templates'] = templates
    context['subdirectory'] = directory_slug
    return render(request, "comps/comp_listing.html", context)