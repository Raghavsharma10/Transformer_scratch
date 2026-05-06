def render_text(self, request, instance, context):
        """
        Custom rendering function for HTML output
        """
        render_template = self.get_render_template(request, instance, email_format='text')
        if not render_template:
            # If there is no TEXT variation, create it by removing the HTML tags.
            base_url = request.build_absolute_uri('/')
            html = self.render_html(request, instance, context)
            return html_to_text(html, base_url)

        instance_context = self.get_context(request, instance, email_format='text', parent_context=context)
        instance_context['email_format'] = 'text'

        text = self.render_to_string(request, render_template, instance_context)
        text = text + ""  # Avoid being a safestring
        if self.render_replace_context_fields:
            text = replace_fields(text, instance_context, autoescape=False)
        return text