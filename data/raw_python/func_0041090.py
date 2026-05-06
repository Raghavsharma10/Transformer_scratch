def delete_member(self, user):
        """Returns a response after attempting to remove
        a member from the list.
        """
        if not self.email_enabled:
            raise EmailNotEnabledError("See settings.EMAIL_ENABLED")
        return requests.delete(
            f"{self.api_url}/{self.address}/members/{user.email}",
            auth=("api", self.api_key),
        )