def email_confirm(request, confirmation_key,
                  template_name='accounts/email_confirm_fail.html',
                  success_url=None, extra_context=None):
    """
    Confirms an email address with a confirmation key.

    Confirms a new email address by running :func:`User.objects.confirm_email`
    method. If the method returns an :class:`User` the user will have his new
    e-mail address set and redirected to ``success_url``. If no ``User`` is
    returned the user will be represented with a fail message from
    ``template_name``.

    :param confirmation_key:
        String with a SHA1 representing the confirmation key used to verify a
        new email address.

    :param template_name:
        String containing the template name which should be rendered when
        confirmation fails. When confirmation is successful, no template is
        needed because the user will be redirected to ``success_url``.

    :param success_url:
        String containing the URL which is redirected to after a successful
        confirmation.  Supplied argument must be able to be rendered by
        ``reverse`` function.

    :param extra_context:
        Dictionary of variables that are passed on to the template supplied by
        ``template_name``.

    """
    user = AccountsSignup.objects.confirm_email(confirmation_key)
    if user:
        if accounts_settings.ACCOUNTS_USE_MESSAGES:
            messages.success(request,
                             _('Your email address has been changed.'),
                             fail_silently=True)

        if success_url:
            redirect_to = success_url
        else:
            redirect_to = reverse('accounts_email_confirm_complete',
                                    kwargs={'username': user.username})
        return redirect(redirect_to)
    else:
        if not extra_context:
            extra_context = dict()
        return ExtraContextTemplateView.as_view(template_name=template_name,
                                        extra_context=extra_context)(request)