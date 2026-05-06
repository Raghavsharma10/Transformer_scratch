async def handle_action(self, action_type, payload, **kwds):
        """
            The default action Handler has no action.
        """
        # if there is a service attached to the action handler
        if hasattr(self, 'service'):
            # handle roll calls
            await roll_call_handler(self.service, action_type, payload, **kwds)