def add_serviceListener(self, type, listener):
        """Adds a listener for a particular service type.  This object
        will then have its update_record method called when information
        arrives for that type."""
        self.remove_service_listener(listener)
        self.browsers.append(ServiceBrowser(self, type, listener))