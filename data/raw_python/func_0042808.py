def account_delete(request, username,
        template_name=accounts_settings.ACCOUNTS_PROFILE_DETAIL_TEMPLATE,
        extra_context=None, **kwargs):
    """
    Delete an account.
    """
    user = get_object_or_404(get_user_model(),
                             username__iexact=username)
    user.is_active = False
    user.save()

    return redirect(reverse('accounts_admin'))