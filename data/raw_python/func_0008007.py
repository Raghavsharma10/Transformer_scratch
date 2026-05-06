def render(self, context=None, clean=False):
        """
        Render email with provided context

        Arguments
        ---------
        context : dict
            |context| If not specified then the
            :attr:`~mail_templated.EmailMessage.context` property is
            used.

        Keyword Arguments
        -----------------
        clean : bool
            If ``True``, remove any template specific properties from the
            message object. Default is ``False``.
        """
        # Load template if it is not loaded yet.
        if not self.template:
            self.load_template(self.template_name)
        # The signature of the `render()` method was changed in Django 1.7.
        # https://docs.djangoproject.com/en/1.8/ref/templates/upgrading/#get-template-and-select-template
        if hasattr(self.template, 'template'):
            context = (context or self.context).copy()
        else:
            context = Context(context or self.context)
        # Add tag strings to the context.
        context.update(self.extra_context)
        result = self.template.render(context)
        # Don't overwrite default value with empty one.
        subject = self._get_block(result, 'subject')
        if subject:
            self.subject = self._get_block(result, 'subject')
        body = self._get_block(result, 'body')
        is_html_body = False
        # The html block is optional, and it also may be set manually.
        html = self._get_block(result, 'html')
        if html:
            if not body:
                # This is an html message without plain text part.
                body = html
                is_html_body = True
            else:
                # Add alternative content.
                self.attach_alternative(html, 'text/html')
        # Don't overwrite default value with empty one.
        if body:
            self.body = body
            if is_html_body:
                self.content_subtype = 'html'
        self._is_rendered = True
        if clean:
            self.clean()