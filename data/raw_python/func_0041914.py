def update_notification_list(self, apps=None, schema_editor=None, verbose=False):
        """Updates the notification model to ensure all registered
        notifications classes are listed.

        Typically called from a post_migrate signal.

        Also, in tests you can register a notification and the Notification
        class (not model) will automatically call this method if the
        named notification does not exist. See notification.notify()
        """
        Notification = (apps or django_apps).get_model("edc_notification.notification")

        # flag all notifications as disabled and re-enable as required
        Notification.objects.all().update(enabled=False)
        if site_notifications.loaded:
            if verbose:
                sys.stdout.write(
                    style.MIGRATE_HEADING("Populating Notification model:\n")
                )
            self.delete_unregistered_notifications(apps=apps)
            for name, notification_cls in site_notifications.registry.items():
                if verbose:
                    sys.stdout.write(
                        f"  * Adding '{name}': '{notification_cls().display_name}'\n"
                    )
                try:
                    obj = Notification.objects.get(name=name)
                except ObjectDoesNotExist:
                    Notification.objects.create(
                        name=name,
                        display_name=notification_cls().display_name,
                        enabled=True,
                    )
                else:
                    obj.display_name = notification_cls().display_name
                    obj.enabled = True
                    obj.save()