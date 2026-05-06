def email_change(request, username, email_form=ChangeEmailForm,
                 template_name='accounts/email_form.html', success_url=None,
                 extra_context=None):
    """
    Change email address

    :param username:
        String of the username which specifies the current account.

    :param email_form:
        Form that will be used to change the email address. Defaults to
        :class:`ChangeEmailForm` supplied by accounts.

    :param template_name:
        String containing the template to be used to display the email form.
        Defaults to ``accounts/email_form.html``.

    :param success_url:
        Named URL where the user will get redirected to when successfully
        changing their email address.  When not supplied will redirect to
        ``accounts_email_complete`` URL.

    :param extra_context:
        Dictionary containing extra variables that can be used to render the
        template. The ``form`` key is always the form supplied by the keyword
        argument ``form`` and the ``user`` key by the user whose email address
        is being changed.

    **Context**

    ``form``
        Form that is used to change the email address supplied by ``form``.

    ``account``
        Instance of the ``Account`` whose email address is about to be changed.

    **Todo**

    Need to have per-object permissions, which enables users with the correct
    permissions to alter the email address of others.

    """
    user = get_object_or_404(get_user_model(), username__iexact=username)

    form = email_form(user)

    if request.method == 'POST':
        form = email_form(user,
                               request.POST,
                               request.FILES)

        if form.is_valid():
            form.save()

            if success_url:
                redirect_to = success_url
            else:
                redirect_to = reverse('accounts_email_change_complete',
                                        kwargs={'username': user.username})
            return redirect(redirect_to)

    if not extra_context:
        extra_context = dict()
    extra_context['form'] = form
    extra_context['profile'] = user.get_profile()
    return ExtraContextTemplateView.as_view(template_name=template_name,
                                        extra_context=extra_context)(request)