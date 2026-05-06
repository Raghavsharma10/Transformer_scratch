def create_mailing_lists(self, verbose=True):
        """Creates the mailing list for each registered notification.
        """
        responses = {}
        if (
            settings.EMAIL_ENABLED
            and self.loaded
            and settings.EMAIL_BACKEND
            != "django.core.mail.backends.locmem.EmailBackend"
        ):
            sys.stdout.write(style.MIGRATE_HEADING(f"Creating mailing lists:\n"))
            for name, notification_cls in self.registry.items():
                message = None
                notification = notification_cls()
                manager = MailingListManager(
                    address=notification.email_to,
                    name=notification.name,
                    display_name=notification.display_name,
                )
                try:
                    response = manager.create()
                except ConnectionError as e:
                    sys.stdout.write(
                        style.ERROR(
                            f"  * Failed to create mailing list {name}. " f"Got {e}\n"
                        )
                    )
                else:
                    if verbose:
                        try:
                            message = response.json().get("message")
                        except JSONDecodeError:
                            message = response.text
                        sys.stdout.write(
                            f"  * Creating mailing list {name}. "
                            f'Got {response.status_code}: "{message}"\n'
                        )
                    responses.update({name: response})
        return responses