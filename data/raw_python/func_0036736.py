def on_discovery_update(self, name, new_config):
        """
        Once a Discovery is updated we update each associated Service to reset
        its up/down status so that the next iteration of the `check_loop`
        loop does the proper reporting again.
        """
        for service in self.configurables[Service].values():
            if service.discovery == name:
                service.reset_status()