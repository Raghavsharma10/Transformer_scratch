def update_record(self, zeroconf, now, record):
        """Callback invoked by Zeroconf when new information arrives.

        Updates information required by browser in the Zeroconf cache."""
        if record.type == _TYPE_PTR and record.name == self.type:
            expired = record.is_expired(now)
            try:
                oldrecord = self.services[record.alias.lower()]
                if not expired:
                    oldrecord.reset_ttl(record)
                else:
                    del(self.services[record.alias.lower()])
                    callback = lambda x: self.listener.remove_service(x,
                            self.type, record.alias)
                    self.list.append(callback)
                    return
            except:
                if not expired:
                    self.services[record.alias.lower()] = record
                    callback = lambda x: self.listener.add_service(x,
                            self.type, record.alias)
                    self.list.append(callback)

            expires = record.get_expiration_time(75)
            if expires < self.next_time:
                self.next_time = expires