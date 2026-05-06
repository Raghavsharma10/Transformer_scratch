def remove_all_connections(provider_id):
    """Remove all connections for the authenticated user to the
    specified provider
    """
    provider = get_provider_or_404(provider_id)

    ctx = dict(provider=provider.name, user=current_user)

    deleted = _datastore.delete_connections(user_id=current_user.get_id(),
                                            provider_id=provider_id)
    if deleted:
        after_this_request(_commit)
        msg = ('All connections to %s removed' % provider.name, 'info')
        connection_removed.send(current_app._get_current_object(),
                                user=current_user._get_current_object(),
                                provider_id=provider_id)
    else:
        msg = ('Unable to remove connection to %(provider)s' % ctx, 'error')

    do_flash(*msg)
    return redirect(request.referrer)