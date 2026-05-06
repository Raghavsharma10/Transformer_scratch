def clean_email(self):
        """ Raise ValidationError if the contact exists. """
        contacts = self.api.lists.contacts(id=self.list_id)['result']

        for contact in contacts:
            if contact['email'] == self.cleaned_data['email']:
                raise forms.ValidationError(
                    _(u'This email is already subscribed'))

        return self.cleaned_data['email']