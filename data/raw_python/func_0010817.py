def get_template_names(self):
        """
        Returns the name of the template to use to render this request.

        Smartmin provides default templates as fallbacks, so appends it's own templates names to the end
        of whatever list is built by the generic views.

        Subclasses can override this by setting a 'template_name' variable on the class.
        """
        templates = []
        if getattr(self, 'template_name', None):
            templates.append(self.template_name)

        if getattr(self, 'default_template', None):
            templates.append(self.default_template)
        else:
            templates = super(SmartView, self).get_template_names()

        return templates