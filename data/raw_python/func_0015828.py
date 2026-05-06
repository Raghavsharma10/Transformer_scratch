def get_context_data(self, **kwargs):
        """
        Returns context dictionary for view.

        :rtype: dict.
        """
        kwargs.update({
            'view':             self,
            'email_form':       EmailLinkForm(),
            'external_form':    ExternalLinkForm(),
            'type_email':       Link.LINK_TYPE_EMAIL,
            'type_external':    Link.LINK_TYPE_EXTERNAL,
        })

        # If a form has been submitted, update context with
        # the submitted form value.
        if 'form' in kwargs:
            submitted_form = kwargs.pop('form')
            if isinstance(submitted_form, EmailLinkForm):
                kwargs.update({'email_form': submitted_form})
            elif isinstance(submitted_form, ExternalLinkForm):
                kwargs.update({'external_form': submitted_form})

        return kwargs