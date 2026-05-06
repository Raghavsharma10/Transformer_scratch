def unsubscribe(self, user, verbose=None):
        """Returns a response after attempting to unsubscribe
        a member from the list.
        """
        if not self.email_enabled:
            raise EmailNotEnabledError("See settings.EMAIL_ENABLED")
        response = requests.put(
            f"{self.api_url}/{self.address}/members/{user.email}",
            auth=("api", self.api_key),
            data={"subscribed": False},
        )
        if verbose:
            sys.stdout.write(
                f"Unsubscribing {user.email} from {self.address}. "
                f"Got response={response.status_code}.\n"
            )
        return response