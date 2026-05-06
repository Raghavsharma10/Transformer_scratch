def update_webhook(self, webhook_url, webhook_id, events=None):
        """Register webhook (if it doesn't exit)."""
        hooks = self._request(MINUT_WEBHOOKS_URL, request_type='GET')['hooks']
        try:
            self._webhook = next(
                hook for hook in hooks if hook['url'] == webhook_url)
            _LOGGER.debug("Webhook: %s", self._webhook)
        except StopIteration:  # Not found
            if events is None:
                events = [e for v in EVENTS.values() for e in v if e]
            self._webhook = self._register_webhook(webhook_url, events)
            _LOGGER.debug("Registered hook: %s", self._webhook)
            return self._webhook