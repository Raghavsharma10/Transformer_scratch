def create(self, verbose=None):
        """Returns a response after attempting to create the list.
        """
        if not self.email_enabled:
            raise EmailNotEnabledError("See settings.EMAIL_ENABLED")
        response = requests.post(
            self.api_url,
            auth=("api", self.api_key),
            data={
                "address": self.address,
                "name": self.name,
                "description": self.display_name,
            },
        )
        if verbose:
            sys.stdout.write(
                f"Creating mailing list {self.address}. "
                f"Got response={response.status_code}.\n"
            )
        return response