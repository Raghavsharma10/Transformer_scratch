def add_custom_nl2br_filters(self, app):
        """Add a custom filter nl2br to jinja2
         Replaces all newline to <BR>
        """
        _paragraph_re = re.compile(r'(?:\r\n|\r|\n){3,}')

        @app.template_filter()
        @evalcontextfilter
        def nl2br(eval_ctx, value):
            result = '\n\n'.join('%s' % p.replace('\n', '<br>\n')
                                 for p in _paragraph_re.split(value))
            return result