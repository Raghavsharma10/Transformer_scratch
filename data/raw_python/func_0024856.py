def _generate_publish_headers(self):
        """
        generate the headers for the connection to event hub service based on the provided config
        :return: {} headers
        """
        headers = {
            'predix-zone-id': self.eventhub_client.zone_id
        }
        token = self.eventhub_client.service._get_bearer_token()
        if self.config.is_grpc():
            headers['authorization'] = token[(token.index(' ') + 1):]
        else:
            headers['authorization'] = token

        if self.config.topic == '':
            headers['topic'] = self.eventhub_client.zone_id + '_topic'
        else:
            headers['topic'] = self.config.topic

        if self.config.publish_type == self.config.Type.SYNC:
            headers['sync-acks'] = 'true'
        else:
            headers['sync-acks'] = 'false'
            headers['send-acks-interval'] = str(self.config.async_cache_ack_interval_millis)
            headers['acks'] = str(self.config.async_enable_acks).lower()
            headers['nacks'] = str(self.config.async_enable_nacks_only).lower()
            headers['cache-acks'] = str(self.config.async_cache_acks_and_nacks).lower()
        return headers