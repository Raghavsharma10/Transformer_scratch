def trigger_event(self, event, client, args, force_dispatch=False):
        """Trigger a new event that will be dispatched to all modules."""
        self.controller.process_event(event, client, args, force_dispatch=force_dispatch)