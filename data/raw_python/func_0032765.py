def add_badge_roles(app):
    """Add ``badge`` role to your sphinx documents. It can create
    a colorful badge inline.
    """
    from docutils.nodes import inline, make_id
    from docutils.parsers.rst.roles import set_classes

    def create_badge_role(color=None):
        def badge_role(name, rawtext, text, lineno, inliner,
                       options=None, content=None):
            options = options or {}
            set_classes(options)
            classes = ['badge']
            if color is None:
                classes.append('badge-' + make_id(text))
            else:
                classes.append('badge-' + color)
            if len(text) == 1:
                classes.append('badge-one')
            options['classes'] = classes
            node = inline(rawtext, text, **options)
            return [node], []
        return badge_role

    app.add_role('badge', create_badge_role())
    app.add_role('badge-red', create_badge_role('red'))
    app.add_role('badge-blue', create_badge_role('blue'))
    app.add_role('badge-green', create_badge_role('green'))
    app.add_role('badge-yellow', create_badge_role('yellow'))