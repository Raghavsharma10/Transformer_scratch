def get_eventhub_host(self):
        """
        returns the publish grpc endpoint for ingestion.
        """
        for protocol in self.service.settings.data['publish']['protocol_details']:
            if protocol['protocol'] == 'grpc':
                return protocol['uri'][0:protocol['uri'].index(':')]