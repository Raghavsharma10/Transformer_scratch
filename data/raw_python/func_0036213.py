def receive(self, event_type, signature, data_str):
        """Receive a web hook for the event and signature.

        Args:
            event_type (str): Name of the event that was received (from the
                request ``X-HelpScout-Event`` header).
            signature (str): The signature that was received, which serves as
                authentication (from the request ``X-HelpScout-Signature``
                header).
            data_str (str): The raw data that was posted by HelpScout
                to the web hook. This must be the raw string, because if it
                is parsed with JSON it will lose its ordering and not pass
                signature validation.

        Raises:
            helpscout.exceptions.HelpScoutSecurityException: If an invalid
                signature is provided, and ``raise_if_invalid`` is ``True``.

        Returns:
            helpscout.web_hook.WebHookEvent: The authenticated web hook
                request.
        """

        if not self.validate_signature(signature, data_str):
            raise HelpScoutSecurityException(
                'The signature provided by this request was invalid.',
            )

        return HelpScoutWebHookEvent(
            event_type=event_type,
            record=json.loads(data_str),
        )