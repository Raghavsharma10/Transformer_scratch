async def announce(self):
        """
            This method is used to announce the existence of the service
        """
        # send a serialized event
        await self.event_broker.send(
            action_type=intialize_service_action(),
            payload=json.dumps(self.summarize())
        )