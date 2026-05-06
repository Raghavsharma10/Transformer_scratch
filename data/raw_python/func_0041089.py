def delete(self):
        """Returns a response after attempting to delete the list.
        """
        if not self.email_enabled:
            raise EmailNotEnabledError("See settings.EMAIL_ENABLED")
        return requests.delete(
            f"{self.api_url}/{self.address}", auth=("api", self.api_key)
        )