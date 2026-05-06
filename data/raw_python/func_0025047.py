def _generate_subscribe_headers(self):
        """
        generate the subscribe stub headers based on the supplied config
        :return: i
        """
        headers =[]
        headers.append(('predix-zone-id', self.eventhub_client.zone_id))

        token = self.eventhub_client.service._get_bearer_token()
        headers.append(('subscribername', self._config.subscriber_name))
        headers.append(('authorization', token[(token.index(' ') + 1):]))

        if self._config.topics is []:
            headers.append(('topic', self.eventhub_client.zone_id + '_topic'))
        else:
            for topic in self._config.topics:
                headers.append(('topic', topic))

        headers.append(('offset-newest', str(self._config.recency == self._config.Recency.NEWEST).lower()))

        headers.append(('acks', str(self._config.acks_enabled).lower()))
        if self._config.acks_enabled:
            headers.append(('max-retries', str(self._config.ack_max_retries)))
            headers.append(('retry-interval', str(self._config.ack_retry_interval_seconds) + 's'))
            headers.append(('duration-before-retry', str(self._config.ack_duration_before_retry_seconds) + 's'))

        if self._config.batching_enabled:
            headers.append(('batch-size', str(self._config.batch_size)))
            headers.append(('batch-interval', str(self._config.batch_interval_millis) + 'ms'))

        return headers