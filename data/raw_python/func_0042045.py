def render_html(self, request, instance, context):
        """
        Custom rendering function for HTML output
        """
        render_template = self.get_render_template(request, instance, email_format='html')
        if not render_template:
            return str(u"{No HTML rendering defined for class '%s'}" % self.__class__.__name__)

        instance_context = self.get_context(request, instance, email_format='html', parent_context=context)
        instance_context['email_format'] = 'html'

        html = self.render_to_string(request, render_template, instance_context)
        if self.render_replace_context_fields:
            html = replace_fields(html, instance_context)  # pass safe-string
        return html