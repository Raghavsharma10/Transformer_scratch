def calls(self):
        """
        Provides access to call overview for the given webhook.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/webhook-calls

        :return: :class:`WebhookWebhooksCallProxy <contentful_management.webhook_webhooks_call_proxy.WebhookWebhooksCallProxy>` object.
        :rtype: contentful.webhook_webhooks_call_proxy.WebhookWebhooksCallProxy

        Usage:

            >>> webhook_webhooks_call_proxy = webhook.calls()
            <WebhookWebhooksCallProxy space_id="cfexampleapi" webhook_id="my_webhook">
        """
        return WebhookWebhooksCallProxy(self._client, self.sys['space'].id, self.sys['id'])