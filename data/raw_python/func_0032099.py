def _get_template(self, template_name):
        """
        Retrieve the cached version of the template
        """
        if template_name not in self.chached_templates:
            self.chached_templates[template_name] = self.env.get_template(template_name)
        return self.chached_templates[template_name]