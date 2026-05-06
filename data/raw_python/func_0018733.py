def update_record(self, zeroconf, now, record):
        """Updates service information from a DNS record"""
        if record is not None and not record.is_expired(now):
            if record.type == _TYPE_A:
                if record.name == self.name:
                    if not record.address in self.address:
                        self.address.append(record.address)
            elif record.type == _TYPE_SRV:
                if record.name == self.name:
                    self.server = record.server
                    self.port = record.port
                    self.weight = record.weight
                    self.priority = record.priority
                    self.address = []
                    self.update_record(zeroconf, now,
                            zeroconf.cache.get_by_details(self.server,
                                _TYPE_A, _CLASS_IN))
            elif record.type == _TYPE_TXT:
                if record.name == self.name:
                    self.set_text(record.text)