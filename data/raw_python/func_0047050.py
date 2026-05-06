def process_event(self, event, client, args, force_dispatch=False):
        """Process an incoming event.

        Offers it to each module according to self.module_ordering,
        continuing to the next unless the module inhibits propagation.

        Returns True if a module inhibited propagation, otherwise False.
        """
        if not self.running:
            _log.debug("Ignoring '%s' event - controller not running.", event)
            return

        # We keep a copy of the state of loaded modules before this event,
        # and restore it when we're done. This lets us handle events that
        # result in other events being dispatched in a graceful manner.
        old_loaded = self.loaded_on_this_event
        self.loaded_on_this_event = set(old_loaded or []) if not force_dispatch else set()

        try:
            _log.debug("Controller is dispatching '%s' event", event)
            for module_name in self.module_ordering:
                if module_name in self.loaded_on_this_event and not force_dispatch:
                    _log.debug("Not dispatching %s to '%s' because it was just "
                               "loaded (%r).", event, module_name,
                               self.loaded_on_this_event)
                    continue
                module = self.loaded_modules[module_name]
                if module.handle_event(event, client, args):
                    return True
        finally:
            self.loaded_on_this_event = old_loaded