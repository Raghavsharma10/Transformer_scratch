def notification_on_post_create_historical_record(
    instance, history_date, history_user, history_change_reason, **kwargs
):
    """Checks and processes any notifications for this model.

    Processes if `label_lower` is in site_notifications.models.

    Note, this is the post_create of the historical model.
    """
    if (
        site_notifications.loaded
        and instance._meta.label_lower in site_notifications.models
    ):
        opts = dict(
            instance=instance,
            user=instance.user_modified or instance.user_created,
            history_date=history_date,
            history_user=history_user,
            history_change_reason=history_change_reason,
            fail_silently=True,
            **kwargs
        )
        site_notifications.notify(**opts)