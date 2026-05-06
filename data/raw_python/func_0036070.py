def create(cls, session, web_hook):
        """Create a web hook.

        Note that creating a new web hook will overwrite the web hook that is
        already configured for this company. There is also no way to
        programmatically determine if a web hook already exists for the
        company. This is a limitation of the HelpScout API and cannot be
        circumvented.

        Args:
            session (requests.sessions.Session): Authenticated session.
            web_hook (helpscout.models.WebHook): The web hook to be created.

        Returns:
            bool: ``True`` if the creation was a success. Errors otherwise.
        """
        cls(
            '/hooks.json',
            data=web_hook.to_api(),
            request_type=RequestPaginator.POST,
            session=session,
        )
        return True