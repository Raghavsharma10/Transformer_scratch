def _read_services(self, services):
        """Get actions from services."""
        for service in services:
            parser = FritzSCDPParser(self.address, self.port, service)
            actions = parser.get_actions()
            service.actions = {action.name: action for action in actions}
            self.services[service.name] = service