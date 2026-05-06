def _register_webhook(self, webhook_url, events):
        """Register webhook."""
        response = self._request(
            MINUT_WEBHOOKS_URL,
            request_type='POST',
            json={
                'url': webhook_url,
                'events': events,
            },
        )
        return response