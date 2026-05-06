def delete_unregistered_notifications(self, apps=None):
        """Delete orphaned notification model instances.
        """
        Notification = (apps or django_apps).get_model("edc_notification.notification")
        return Notification.objects.exclude(
            name__in=[n.name for n in site_notifications.registry.values()]
        ).delete()