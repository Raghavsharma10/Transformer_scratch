def html(self, groups='all', template=None, **context):
        """Return an html string of the routes specified by the doc() method

        A template can be specified. A list of routes is available under the
        'autodoc' value (refer to the documentation for the generate() for a
        description of available values). If no template is specified, a
        default template is used.

        By specifying the group or groups arguments, only routes belonging to
        those groups will be returned.
        """
        context['autodoc'] = context['autodoc'] if 'autodoc' in context \
            else self.generate(groups=groups)
        context['defaults'] = context['defaults'] if 'defaults' in context \
            else self.default_props
        if template:
            return render_template(template, **context)
        else:
            filename = os.path.join(
                os.path.dirname(__file__),
                'templates',
                'autodoc_default.html'
            )
            with open(filename) as file:
                content = file.read()
                with current_app.app_context():
                    return render_template_string(content, **context)