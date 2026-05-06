def to_json(self):
        """
        Returns the JSON representation of the webhook.
        """

        result = super(Webhook, self).to_json()
        result.update({
            'name': self.name,
            'url': self.url,
            'topics': self.topics,
            'httpBasicUsername': self.http_basic_username,
            'headers': self.headers
        })

        if self.filters:
            result.update({'filters': self.filters})

        if self.transformation:
            result.update({'transformation': self.transformation})

        return result