def start(self):
        """Begin listening for events from the Client and acting upon them.

        Note: If configuration has not already been loaded, it will be loaded
        immediately before starting to listen for events. Calling this method
        without having specified and/or loaded a configuration will result in
        completely default values being used.

        After all modules for this controller are loaded, the STARTUP event
        will be dispatched.
        """
        if not self.config and self.config_path is not None:
            self.load_config()
        self.running = True
        self.process_event("STARTUP", self.client, ())