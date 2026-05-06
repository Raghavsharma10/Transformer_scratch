def activate(request, activation_key,
             template_name='accounts/activate_fail.html',
             success_url=None, extra_context=None):
    """
    Activate a user with an activation key.

    The key is a SHA1 string. When the SHA1 is found with an
    :class:`AccountsSignup`, the :class:`User` of that account will be
    activated.  After a successful activation the view will redirect to
    ``success_url``.  If the SHA1 is not found, the user will be shown the
    ``template_name`` template displaying a fail message.

    :param activation_key:
        String of a SHA1 string of 40 characters long. A SHA1 is always 160bit
        long, with 4 bits per character this makes it --160/4-- 40 characters
        long.

    :param template_name:
        String containing the template name that is used when the
        ``activation_key`` is invalid and the activation fails. Defaults to
        ``accounts/activation_fail.html``.

    :param success_url:
        String containing the URL where the user should be redirected to after
        a successful activation. Will replace ``%(username)s`` with string
        formatting if supplied. If ``success_url`` is left empty, will direct
        to ``accounts_profile_detail`` view.

    :param extra_context:
        Dictionary containing variables which could be added to the template
        context. Default to an empty dictionary.

    """
    user = AccountsSignup.objects.activate_user(activation_key)
    if user:
        # Sign the user in.
        auth_user = authenticate(identification=user.email,
                                 check_password=False)
        login(request, auth_user)

        if accounts_settings.ACCOUNTS_USE_MESSAGES:
            messages.success(request,
                             _('Your account has been activated and you have been signed in.'),
                             fail_silently=True)

        if success_url:
            redirect_to = success_url % {'username': user.username}
        else:
            redirect_to = reverse('accounts_profile_detail',
                                    kwargs={'username': user.username})
        return redirect(redirect_to)
    else:
        if not extra_context:
            extra_context = dict()
        return ExtraContextTemplateView.as_view(template_name=template_name,
                                        extra_context=extra_context)(request)