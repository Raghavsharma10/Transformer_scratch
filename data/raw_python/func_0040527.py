def manage_mailists_on_userprofile_m2m_changed(
    action, instance, pk_set, sender, **kwargs
):
    """Updates the mail server mailing lists based on the
    selections in the UserProfile model.
    """
    try:
        instance.email_notifications
    except AttributeError:
        pass
    else:
        if action == "post_remove":
            update_mailing_lists_in_m2m(
                sender=sender,
                userprofile=instance,
                unsubscribe=True,
                pk_set=pk_set,
                verbose=True,
            )
        elif action == "post_add":
            update_mailing_lists_in_m2m(
                sender=sender,
                userprofile=instance,
                subscribe=True,
                pk_set=pk_set,
                verbose=True,
            )