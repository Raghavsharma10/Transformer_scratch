def trigger(self, event, *args):
        """Triggers the specified event by invoking EventHook.trigger under the hood.

        @param event: event to trigger. Any object can be passed
                      as event, but string is preferable. If qcore.EnumBase
                      instance is passed, its name is used as event key.
        @param args: event arguments.
        @return: self, so calls like this can be chained together.

        """
        event_hook = self.get_or_create(event)
        event_hook.trigger(*args)
        return self