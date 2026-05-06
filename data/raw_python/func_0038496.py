def render_page(path):
    """Internal interface to the page view.

    :param path: Page path.
    :returns: The rendered template.
    """
    try:
        page = Page.get_by_url(request.path)
    except NoResultFound:
        abort(404)

    return render_template(
        [page.template_name, current_app.config['PAGES_DEFAULT_TEMPLATE']],
        page=page)