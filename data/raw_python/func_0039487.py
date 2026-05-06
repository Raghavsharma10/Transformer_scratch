def register(**kwargs):
    """Registers a notification_cls.
    """

    def _wrapper(notification_cls):

        if not issubclass(notification_cls, (Notification,)):
            raise RegisterNotificationError(
                f"Wrapped class must be a 'Notification' class. "
                f"Got '{notification_cls.__name__}'"
            )
        site_notifications.register(notification_cls=notification_cls)
        return notification_cls

    return _wrapper