def subscribe(self, user, verbose=None):
        """Returns a response after attempting to subscribe
        a member to the list.
        """
        if not self.email_enabled:
            raise EmailNotEnabledError("See settings.EMAIL_ENABLED")
        if not user.email:
            raise UserEmailError(f"User {user}'s email address is not defined.")
        response = requests.post(
            f"{self.api_url}/{self.address}/members",
            auth=("api", self.api_key),
            data={
                "subscribed": True,
                "address": user.email,
                "name": f"{user.first_name} {user.last_name}",
                "description": f'{user.userprofile.job_title or ""}',
                "upsert": "yes",
            },
        )
        if verbose:
            sys.stdout.write(
                f"Subscribing {user.email} to {self.address}. "
                f"Got response={response.status_code}.\n"
            )
        return response