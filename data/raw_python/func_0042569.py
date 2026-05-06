def clean_email(self):
        """
        Validate that the e-mail address is unique.
        """
        if get_user_model().objects.filter(
            email__iexact=self.cleaned_data['email']):
            raise forms.ValidationError(_('This email is already in use. Please supply a different email.'))
        return self.cleaned_data['email']