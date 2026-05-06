def create_eventhub(self, **kwargs):
        """
        todo make it so the client can be customised to publish/subscribe
        Creates an instance of eventhub service
        """
        eventhub = predix.admin.eventhub.EventHub(**kwargs)
        eventhub.create()
        eventhub.add_to_manifest(self)
        eventhub.grant_client(client_id=self.get_client_id(), **kwargs)
        eventhub.add_to_manifest(self)
        return eventhub