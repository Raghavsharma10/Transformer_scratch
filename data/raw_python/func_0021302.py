def is_fresh(self, access_mode):
        """ Indicator for whether routing information is still usable.
        """
        log_debug("[#0000]  C: <ROUTING> Checking table freshness for %r", access_mode)
        expired = self.last_updated_time + self.ttl <= self.timer()
        has_server_for_mode = bool(access_mode == READ_ACCESS and self.readers) or bool(access_mode == WRITE_ACCESS and self.writers)
        log_debug("[#0000]  C: <ROUTING> Table expired=%r", expired)
        log_debug("[#0000]  C: <ROUTING> Table routers=%r", self.routers)
        log_debug("[#0000]  C: <ROUTING> Table has_server_for_mode=%r", has_server_for_mode)
        return not expired and self.routers and has_server_for_mode