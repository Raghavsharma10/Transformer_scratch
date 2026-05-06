def notification_model(self):
        """Returns the Notification 'model' instance associated
        with this notification.
        """
        NotificationModel = django_apps.get_model("edc_notification.notification")
        # trigger exception if this class is not registered.
        site_notifications.get(self.name)
        try:
            notification_model = NotificationModel.objects.get(name=self.name)
        except ObjectDoesNotExist:
            site_notifications.update_notification_list()
            notification_model = NotificationModel.objects.get(name=self.name)
        return notification_model