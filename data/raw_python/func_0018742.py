def check_service(self, info):
        """Checks the network for a unique service name, modifying the
        ServiceInfo passed in if it is not unique."""
        now = current_time_millis()
        next_time = now
        i = 0
        while i < 3:
            for record in self.cache.entries_with_name(info.type):
                if record.type == _TYPE_PTR and \
                        not record.is_expired(now) and \
                        record.alias == info.name:
                    if (info.name.find('.') < 0):
                        info.name = info.name + ".[" + \
                                info.address + \
                                ":" + info.port + \
                                "]." + info.type
                        self.check_service(info)
                        return
                    raise NonUniqueNameException
            if now < next_time:
                self.wait(next_time - now)
                now = current_time_millis()
                continue
            out = DNSOutgoing(_FLAGS_QR_QUERY | _FLAGS_AA)
            self.debug = out
            out.add_question(
                    DNSQuestion(info.type, _TYPE_PTR, _CLASS_IN))
            out.add_authorative_answer(
                    DNSPointer(info.type,
                        _TYPE_PTR, _CLASS_IN, info.ttl, info.name))
            self.send(out)
            i += 1
            next_time += _CHECK_TIME