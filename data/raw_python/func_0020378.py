def get(self, request, key):
        """Validate an email with the given key"""

        try:
            email_val = EmailAddressValidation.objects.get(validation_key=key)
        except EmailAddressValidation.DoesNotExist:
            messages.error(request, _('The email address you are trying to '
                                      'verify either has already been verified'
                                      ' or does not exist.'))
            return redirect('/')

        try:
            email = EmailAddress.objects.get(address=email_val.address)
        except EmailAddress.DoesNotExist:
            email = EmailAddress(address=email_val.address)

        if email.user and email.user.is_active:
            messages.error(request, _('The email address you are trying to '
                                      'verify is already an active email '
                                      'address.'))
            email_val.delete()
            return redirect('/')

        email.user = email_val.user
        email.save()
        email_val.delete()

        user = User.objects.get(username=email.user.username)
        user.is_active = True
        user.save()

        messages.success(request, _('Email address verified!'))
        return redirect('user_profile', username=email_val.user.username)