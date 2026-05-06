def render_to_response(self, context):
        """Add django-crispy form helper and draw the template

        Returns the ``TemplateResponse`` ready to be displayed

        """
        self.setup_forms()
        return TemplateResponse(
            self.request, self.form_template,
            context, current_app=self.admin_site.name)