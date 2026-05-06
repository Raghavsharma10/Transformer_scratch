def signout(request, next_page=accounts_settings.ACCOUNTS_REDIRECT_ON_SIGNOUT,
            template_name='accounts/signout.html', *args, **kwargs):
    """
    Signs out the user and adds a success message ``You have been signed
    out.`` If next_page is defined you will be redirected to the URI. If
    not the template in template_name is used.

    :param next_page:
        A string which specifies the URI to redirect to.

    :param template_name:
        String defining the name of the template to use. Defaults to
        ``accounts/signout.html``.

    """
    if request.user.is_authenticated() and \
            accounts_settings.ACCOUNTS_USE_MESSAGES:  # pragma: no cover
        messages.success(request, _('You have been signed out.'),
                         fail_silently=True)
    return Signout(request, next_page, template_name, *args, **kwargs)