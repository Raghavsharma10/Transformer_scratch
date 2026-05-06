def handle_not_found(exception, **extra):
    """Custom blueprint exception handler."""
    assert isinstance(exception, NotFound)

    page = Page.query.filter(db.or_(Page.url == request.path,
                                    Page.url == request.path + "/")).first()

    if page:
        _add_url_rule(page.url)
        return render_template(
            [
                page.template_name,
                current_app.config['PAGES_DEFAULT_TEMPLATE']
            ],
            page=page
        )
    elif 'wrapped' in extra:
        return extra['wrapped'](exception)
    else:
        return exception