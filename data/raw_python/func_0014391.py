def health(self):
        """
        Provides access to health overview for the given webhook.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/webhook-calls/webhook-health

        :return: :class:`WebhookWebhooksHealthProxy <contentful_management.webhook_webhooks_health_proxy.WebhookWebhooksHealthProxy>` object.
        :rtype: contentful.webhook_webhooks_health_proxy.WebhookWebhooksHealthProxy

        Usage:

            >>> webhook_webhooks_health_proxy = webhook.health()
            <WebhookWebhooksHealthProxy space_id="cfexampleapi" webhook_id="my_webhook">
        """
        return WebhookWebhooksHealthProxy(self._client, self.sys['space'].id, self.sys['id'])