def password_change(request, username,
                    template_name='accounts/password_form.html',
                    pass_form=PasswordChangeForm, success_url=None,
                    extra_context=None):
    """ Change password of user.

    This view is almost a mirror of the view supplied in
    :func:`contrib.auth.views.password_change`, with the minor change that in
    this view we also use the username to change the password. This was needed
    to keep our URLs logical (and REST) across the entire application. And
    that in a later stadium administrators can also change the users password
    through the web application itself.

    :param username:
        String supplying the username of the user who's password is about to be
        changed.

    :param template_name:
        String of the name of the template that is used to display the password
        change form. Defaults to ``accounts/password_form.html``.

    :param pass_form:
        Form used to change password. Default is the form supplied by Django
        itself named ``PasswordChangeForm``.

    :param success_url:
        Named URL that is passed onto a :func:`reverse` function with
        ``username`` of the active user. Defaults to the
        ``accounts_password_complete`` URL.

    :param extra_context:
        Dictionary of extra variables that are passed on to the template. The
        ``form`` key is always used by the form supplied by ``pass_form``.

    **Context**

    ``form``
        Form used to change the password.

    """
    user = get_object_or_404(get_user_model(),
                             username__iexact=username)

    form = pass_form(user=user)

    if request.method == "POST":
        form = pass_form(user=user, data=request.POST)
        if form.is_valid():
            form.save()

            # Send a signal that the password has changed
            accounts_signals.password_complete.send(sender=None,
                                                   user=user)

            if success_url:
                redirect_to = success_url
            else:
                redirect_to = reverse('accounts_password_change_complete',
                                        kwargs={'username': user.username})
            return redirect(redirect_to)

    if not extra_context:
        extra_context = dict()
    extra_context['form'] = form
    extra_context['profile'] = user.get_profile()
    return ExtraContextTemplateView.as_view(template_name=template_name,
                                        extra_context=extra_context)(request)