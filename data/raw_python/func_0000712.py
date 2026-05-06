def remove_connection(provider_id, provider_user_id):
    """Remove a specific connection for the authenticated user to the
    specified provider
    """
    provider = get_provider_or_404(provider_id)

    ctx = dict(provider=provider.name, user=current_user,
               provider_user_id=provider_user_id)

    deleted = _datastore.delete_connection(user_id=current_user.get_id(),
                                           provider_id=provider_id,
                                           provider_user_id=provider_user_id)

    if deleted:
        after_this_request(_commit)
        msg = ('Connection to %(provider)s removed' % ctx, 'info')
        connection_removed.send(current_app._get_current_object(),
                                user=current_user._get_current_object(),
                                provider_id=provider_id)
    else:
        msg = ('Unabled to remove connection to %(provider)s' % ctx, 'error')

    do_flash(*msg)
    return redirect(request.referrer or get_post_login_redirect())