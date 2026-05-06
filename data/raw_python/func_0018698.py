def to_string(self, other):
        """String representation with addtional information"""
        arg = "%s/%s,%s" % (
                self.ttl, self.get_remaining_ttl(current_time_millis()), other)
        return DNSEntry.to_string(self, "record", arg)