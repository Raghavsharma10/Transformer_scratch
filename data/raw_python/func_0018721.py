def write_record(self, record, now):
        """Writes a record (answer, authoritative answer, additional) to
        the packet"""
        self.write_name(record.name)
        self.write_short(record.type)
        if record.unique and self.multicast:
            self.write_short(record.clazz | _CLASS_UNIQUE)
        else:
            self.write_short(record.clazz)
        if now == 0:
            self.write_int(record.ttl)
        else:
            self.write_int(record.get_remaining_ttl(now))
        index = len(self.data)
        # Adjust size for the short we will write before this record
        #
        self.size += 2
        record.write(self)
        self.size -= 2

        length = len(b''.join(self.data[index:]))
        self.insert_short(index, length)