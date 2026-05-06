def update_status(self):
        """Update status informations in tkinter window."""
        try:
            # all this may fail if the connection to the fritzbox is down
            self.update_connection_status()
            self.max_stream_rate.set(self.get_stream_rate_str())
            self.ip.set(self.status.external_ip)
            self.uptime.set(self.status.str_uptime)
            upstream, downstream = self.status.transmission_rate
        except IOError:
            # here we inform the user about being unable to
            # update the status informations
            pass
        else:
            # max_downstream and max_upstream may be zero if the
            # fritzbox is configured as ip-client.
            if self.max_downstream > 0:
                self.in_meter.set_fraction(
                    1.0 * downstream / self.max_downstream)
            if self.max_upstream > 0:
                self.out_meter.set_fraction(1.0 * upstream / self.max_upstream)
            self.update_traffic_info()
        self.after(1000, self.update_status)