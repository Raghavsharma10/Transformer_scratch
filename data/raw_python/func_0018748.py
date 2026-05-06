def close(self):
        """Ends the background threads, and prevent this instance from
        servicing further queries."""
        if globals()['_GLOBAL_DONE'] == 0:
            globals()['_GLOBAL_DONE'] = 1
            self.notify_all()
            self.engine.notify()
            self.unregister_all_services()
            for i in self.intf.values():
                try:
                    # there are cases, when we start mDNS without network
                    i.setsockopt(socket.SOL_IP, socket.IP_DROP_MEMBERSHIP,
                            socket.inet_aton(_MDNS_ADDR) + \
                                    socket.inet_aton('0.0.0.0'))
                except:
                    pass
                i.close()