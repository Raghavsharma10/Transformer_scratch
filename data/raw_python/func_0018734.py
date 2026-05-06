def request(self, zeroconf, timeout):
        """Returns true if the service could be discovered on the
        network, and updates this object with details discovered.
        """
        now = current_time_millis()
        delay = _LISTENER_TIME
        next = now + delay
        last = now + timeout
        result = 0
        try:
            zeroconf.add_listener(self,
                    DNSQuestion(self.name, _TYPE_ANY, _CLASS_IN))
            while self.server is None or \
                    len(self.address) == 0 or \
                    self.text is None:
                if last <= now:
                    return 0
                if next <= now:
                    out = DNSOutgoing(_FLAGS_QR_QUERY)
                    out.add_question(DNSQuestion(self.name,
                        _TYPE_SRV, _CLASS_IN))
                    out.add_answer_at_time(
                            zeroconf.cache.get_by_details(self.name,
                                _TYPE_SRV, _CLASS_IN), now)
                    out.add_question(
                            DNSQuestion(self.name, _TYPE_TXT, _CLASS_IN))
                    out.add_answer_at_time(
                            zeroconf.cache.get_by_details(self.name,
                                _TYPE_TXT, _CLASS_IN), now)
                    if self.server is not None:
                        out.add_question(
                                DNSQuestion(self.server, _TYPE_A, _CLASS_IN))
                        out.add_answer_at_time(
                                zeroconf.cache.get_by_details(self.server,
                                    _TYPE_A, _CLASS_IN), now)
                    zeroconf.send(out)
                    next = now + delay
                    delay = delay * 2

                zeroconf.wait(min(next, last) - now)
                now = current_time_millis()
            result = 1
        finally:
            zeroconf.remove_listener(self)

        return result