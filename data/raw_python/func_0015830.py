def get_form_class(self):
        """
        Returns form class to use in the view.

        :rtype: django.forms.ModelForm.
        """
        if self.object.link_type == Link.LINK_TYPE_EMAIL:
            return EmailLinkForm
        elif self.object.link_type == Link.LINK_TYPE_EXTERNAL:
            return ExternalLinkForm

        return None