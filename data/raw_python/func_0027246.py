def cancel_expired_invitations(invitations=None):
    """
    Invitation lifetime must be specified in Waldur Core settings with parameter
    "INVITATION_LIFETIME". If invitation creation time is less than expiration time, the invitation will set as expired.
    """
    expiration_date = timezone.now() - settings.WALDUR_CORE['INVITATION_LIFETIME']
    if not invitations:
        invitations = models.Invitation.objects.filter(state=models.Invitation.State.PENDING)
    invitations = invitations.filter(created__lte=expiration_date)
    invitations.update(state=models.Invitation.State.EXPIRED)