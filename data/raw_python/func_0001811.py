def _check_email_match(self, user, email):
        """
        DRY helper.

        Requiring the user to specify both username and email will help catch
        certain issues, for example if the expected username has already been
        taken by someone else.
        """
        if user.email != email:
            # The passed email address doesn't match this username's email address.
            # Assume a problem and fail.
            raise CommandError(
                _(
                    'Skipping user "{}" because the specified and existing email '
                    'addresses do not match.'
                ).format(user.username)
            )