def handle_query(self, msg, addr, port, orig):
        """
        Deal with incoming query packets.  Provides a response if
        possible.

        msg    - message to process
        addr    - dst addr
        port    - dst port
        orig    - originating address (for adaptive records)
        """
        out = None

        # Support unicast client responses
        #
        if port != _MDNS_PORT:
            out = DNSOutgoing(_FLAGS_QR_RESPONSE | _FLAGS_AA, 0)
            for question in msg.questions:
                out.add_question(question)
        for question in msg.questions:
            if question.type == _TYPE_PTR:
                for service in self.services.values():
                    if question.name == service.type:
                        # FIXME: sometimes we just not in time filling cache
                        answer = self.cache.get(
                                DNSPointer(service.type,
                                    _TYPE_PTR, _CLASS_IN,
                                    service.ttl, service.name))
                        if out is None and answer is not None:
                            out = DNSOutgoing(_FLAGS_QR_RESPONSE | _FLAGS_AA)
                        if answer:
                            out.add_answer(msg, answer)
            if question.type == _TYPE_AXFR:
                if question.name in list(self.zones.keys()):
                    if out is None:
                        out = DNSOutgoing(_FLAGS_QR_RESPONSE | _FLAGS_AA)
                    for i in self.zones[question.name].services.values():
                        out.add_answer(msg, i)
            else:
                try:
                    if out is None:
                        out = DNSOutgoing(_FLAGS_QR_RESPONSE | _FLAGS_AA)

                    service = self.services.get(question.name.lower(), None)
                    try:
                        rs = service.records
                    except:
                        rs = []

                    # Answer A record queries for any service addresses we know
                    if (question.type == _TYPE_A or \
                            question.type == _TYPE_ANY) \
                            and (_TYPE_A in rs):
                        for service in self.services.values():
                            if service.server == question.name.lower():
                                for i in service.address:
                                    out.add_answer(msg, self.cache.get(
                                        DNSAddress(question.name,
                                            _TYPE_A, _CLASS_IN | _CLASS_UNIQUE,
                                            service.ttl, i)))

                    if not service:
                        continue

                    if (question.type == _TYPE_SRV or \
                            question.type == _TYPE_ANY) and (_TYPE_SRV in rs):
                        out.add_answer(msg, self.cache.get(
                            DNSService(question.name,
                                _TYPE_SRV, _CLASS_IN | _CLASS_UNIQUE,
                                service.ttl, service.priority, service.weight,
                                service.port, service.server)))
                    if (question.type == _TYPE_TXT or \
                            question.type == _TYPE_ANY) and \
                            (_TYPE_TXT in rs):
                        out.add_answer(msg, self.cache.get(
                            DNSText(question.name,
                                _TYPE_TXT, _CLASS_IN | _CLASS_UNIQUE,
                                service.ttl, service.text)))
                    if (question.type == _TYPE_SRV) and (_TYPE_SRV in rs):
                        for i in service.address:
                            out.add_additional_answer(self.cache.get(
                                DNSAddress(service.server,
                                    _TYPE_A, _CLASS_IN | _CLASS_UNIQUE,
                                    service.ttl, i)))
                except:
                    traceback.print_exc()

        if out is not None and out.answers:
            out.id = msg.id
            self.send(out, addr, port)