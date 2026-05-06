def send(self, out, addr=_MDNS_ADDR, port=_MDNS_PORT):
        """Sends an outgoing packet."""
        # This is a quick test to see if we can parse the packets we generate
        #temp = DNSIncoming(out.packet())
        for i in self.intf.values():
            try:
                return i.sendto(out.packet(), 0, (addr, port))
            except:
                traceback.print_exc()
                # Ignore this, it may be a temporary loss of network connection
                return -1