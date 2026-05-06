def accept(self, request, uuid=None):
        """ Accept invitation for current user.

            To replace user's email with email from invitation - add parameter
            'replace_email' to request POST body.
        """
        invitation = self.get_object()

        if invitation.state != models.Invitation.State.PENDING:
            raise ValidationError(_('Only pending invitation can be accepted.'))
        elif invitation.civil_number and invitation.civil_number != request.user.civil_number:
            raise ValidationError(_('User has an invalid civil number.'))

        if invitation.project:
            if invitation.project.has_user(request.user):
                raise ValidationError(_('User already has role within this project.'))
        elif invitation.customer.has_user(request.user):
            raise ValidationError(_('User already has role within this customer.'))

        if settings.WALDUR_CORE['VALIDATE_INVITATION_EMAIL'] and invitation.email != request.user.email:
            raise ValidationError(_('Invitation and user emails mismatch.'))

        replace_email = bool(request.data.get('replace_email'))
        invitation.accept(request.user, replace_email=replace_email)
        return Response({'detail': _('Invitation has been successfully accepted.')},
                        status=status.HTTP_200_OK)