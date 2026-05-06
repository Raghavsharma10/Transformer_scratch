def unregister_service(self, info):
        """Unregister a service."""
        try:
            del(self.services[info.name.lower()])
        except:
            pass
        now = current_time_millis()
        next_time = now
        i = 0
        while i < 3:
            if now < next_time:
                self.wait(next_time - now)
                now = current_time_millis()
                continue
            out = DNSOutgoing(_FLAGS_QR_RESPONSE | _FLAGS_AA)
            out.add_answer_at_time(
                    DNSPointer(info.type,
                        _TYPE_PTR, _CLASS_IN, 0, info.name), 0)
            out.add_answer_at_time(
                    DNSService(info.name,
                        _TYPE_SRV, _CLASS_IN, 0, info.priority,
                        info.weight, info.port, info.name), 0)
            out.add_answer_at_time(
                    DNSText(info.name, _TYPE_TXT, _CLASS_IN, 0, info.text), 0)
            for k in info.address:
                out.add_answer_at_time(
                        DNSAddress(info.server, _TYPE_A, _CLASS_IN, 0, k), 0)
            self.send(out)
            i += 1
            next_time += _UNREGISTER_TIME