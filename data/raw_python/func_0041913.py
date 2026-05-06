def notify(self, instance=None, **kwargs):
        """A wrapper to call notification.notify for each notification
        class associated with the given model instance.

        Returns a dictionary of {notification.name: model, ...}
        including only notifications sent.
        """
        notified = {}
        for notification_cls in self.registry.values():
            notification = notification_cls()
            if notification.notify(instance=instance, **kwargs):
                notified.update({notification_cls.name: instance._meta.label_lower})
        return notified