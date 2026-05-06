def pages():
    """Load pages."""
    p1 = Page(
        url='/example1',
        title='My page with default template',
        description='my description',
        content='hello default page',
        template_name='invenio_pages/default.html',
    )
    p2 = Page(
        url='/example2',
        title='My page with my template',
        description='my description',
        content='hello my page',
        template_name='app/mytemplate.html',
    )
    with db.session.begin_nested():
        db.session.add(p1)
        db.session.add(p2)
    db.session.commit()