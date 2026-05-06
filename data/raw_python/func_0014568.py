def update_body(app, pagename, templatename, context, doctree):
    """
    Add Read the Docs content to Sphinx body content.

    This is the most reliable way to inject our content into the page.
    """

    STATIC_URL = context.get('STATIC_URL', DEFAULT_STATIC_URL)
    online_builders = [
        'readthedocs', 'readthedocsdirhtml', 'readthedocssinglehtml'
    ]
    if app.builder.name == 'readthedocssinglehtmllocalmedia':
        if 'html_theme' in context and context['html_theme'] == 'sphinx_rtd_theme':
            theme_css = '_static/css/theme.css'
        else:
            theme_css = '_static/css/badge_only.css'
    elif app.builder.name in online_builders:
        if 'html_theme' in context and context['html_theme'] == 'sphinx_rtd_theme':
            theme_css = '%scss/sphinx_rtd_theme.css' % STATIC_URL
        else:
            theme_css = '%scss/badge_only.css' % STATIC_URL
    else:
        # Only insert on our HTML builds
        return

    inject_css = True

    # Starting at v0.4.0 of the sphinx theme, the theme CSS should not be injected
    # This decouples the theme CSS (which is versioned independently) from readthedocs.org
    if theme_css.endswith('sphinx_rtd_theme.css'):
        try:
            import sphinx_rtd_theme
            inject_css = LooseVersion(sphinx_rtd_theme.__version__) < LooseVersion('0.4.0')
        except ImportError:
            pass

    if inject_css and theme_css not in app.builder.css_files:
        if sphinx.version_info < (1, 8):
            app.builder.css_files.insert(0, theme_css)
        else:
            app.add_css_file(theme_css)

    # This is monkey patched on the signal because we can't know what the user
    # has done with their `app.builder.templates` before now.

    if not hasattr(app.builder.templates.render, '_patched'):
        # Janky monkey patch of template rendering to add our content
        old_render = app.builder.templates.render

        def rtd_render(self, template, render_context):
            """
            A decorator that renders the content with the users template renderer,
            then adds the Read the Docs HTML content at the end of body.
            """
            # Render Read the Docs content
            template_context = render_context.copy()
            template_context['rtd_css_url'] = '{}css/readthedocs-doc-embed.css'.format(STATIC_URL)
            template_context['rtd_analytics_url'] = '{}javascript/readthedocs-analytics.js'.format(
                STATIC_URL,
            )
            source = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                '_templates',
                'readthedocs-insert.html.tmpl'
            )
            templ = open(source).read()
            rtd_content = app.builder.templates.render_string(templ, template_context)

            # Handle original render function
            content = old_render(template, render_context)
            end_body = content.lower().find('</head>')

            # Insert our content at the end of the body.
            if end_body != -1:
                content = content[:end_body] + rtd_content + "\n" + content[end_body:]
            else:
                log.debug("File doesn't look like HTML. Skipping RTD content addition")

            return content

        rtd_render._patched = True
        app.builder.templates.render = types.MethodType(rtd_render,
                                                        app.builder.templates)