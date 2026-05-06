async def send(self, payload='', action_type='', channel=None, **kwds):
        """
            This method sends a message over the kafka stream.
        """
        # use a custom channel if one was provided
        channel = channel or self.producer_channel

        # serialize the action type for the
        message = serialize_action(action_type=action_type, payload=payload, **kwds)
        # send the message
        return await self._producer.send(channel, message.encode())