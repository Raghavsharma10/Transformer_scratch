def signin(request, auth_form=AuthenticationForm,
           template_name='accounts/signin_form.html',
           redirect_field_name=REDIRECT_FIELD_NAME,
           redirect_signin_function=signin_redirect, extra_context=None):
    """
    Signin using email or username with password.

    Signs a user in by combining email/username with password. If the
    combination is correct and the user :func:`is_active` the
    :func:`redirect_signin_function` is called with the arguments
    ``REDIRECT_FIELD_NAME`` and an instance of the :class:`User` who is is
    trying the login. The returned value of the function will be the URL that
    is redirected to.

    A user can also select to be remembered for ``ACCOUNTS_REMEMBER_DAYS``.

    :param auth_form:
        Form to use for signing the user in. Defaults to the
        :class:`AuthenticationForm` supplied by accounts.

    :param template_name:
        String defining the name of the template to use. Defaults to
        ``accounts/signin_form.html``.

    :param redirect_field_name:
        Form field name which contains the value for a redirect to the
        succeeding page. Defaults to ``next`` and is set in
        ``REDIRECT_FIELD_NAME`` setting.

    :param redirect_signin_function:
        Function which handles the redirect. This functions gets the value of
        ``REDIRECT_FIELD_NAME`` and the :class:`User` who has logged in. It
        must return a string which specifies the URI to redirect to.

    :param extra_context:
        A dictionary containing extra variables that should be passed to the
        rendered template. The ``form`` key is always the ``auth_form``.

    **Context**

    ``form``
        Form used for authentication supplied by ``auth_form``.

    """
    form = auth_form()

    if request.method == 'POST':
        form = auth_form(request.POST, request.FILES)
        if form.is_valid():
            identification = form.cleaned_data['identification']
            password = form.cleaned_data['password']
            remember_me = form.cleaned_data['remember_me']

            user = authenticate(identification=identification,
                                password=password)
            if user.is_active:
                login(request, user)
                if remember_me:
                    request.session.set_expiry(accounts_settings.ACCOUNTS_REMEMBER_ME_DAYS[1] * 86400)
                else:
                    request.session.set_expiry(0)

                if accounts_settings.ACCOUNTS_USE_MESSAGES:
                    messages.success(request, _('You have been signed in.'),
                                     fail_silently=True)

                # Whereto now?
                redirect_to = redirect_signin_function(
                    request.GET.get(redirect_field_name), user)
                return redirect(redirect_to)
            else:
                return redirect(reverse('accounts_disabled',
                                        kwargs={'username': user.username}))

    if not extra_context:
        extra_context = dict()
    extra_context.update({
        'form': form,
        'next': request.GET.get(redirect_field_name),
    })
    return ExtraContextTemplateView.as_view(template_name=template_name,
                                        extra_context=extra_context)(request)