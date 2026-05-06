def subject(self):
        """
        Render the subject of the message to a string.

        """
        template_name = self.subject_template_name() if \
            callable(self.subject_template_name) \
            else self.subject_template_name
        subject = loader.render_to_string(
            template_name, self.get_context(), request=self.request
        )
        return ''.join(subject.splitlines())