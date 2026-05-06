def _on_gateway(self, header, payload, rest, addr):
        """
        Records a discovered gateway, for connecting to later.
        """
        if payload.get('service') == SERVICE_UDP:
            self.gateway = Gateway(addr[0], payload['port'], header.gateway)
            self.gateway_found_event.set()