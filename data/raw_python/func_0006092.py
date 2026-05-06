def message(self):
        """
        Render the body of the message to a string.

        """
        template_name = self.template_name() if \
            callable(self.template_name) \
            else self.template_name
        return loader.render_to_string(
            template_name, self.get_context(), request=self.request
        )