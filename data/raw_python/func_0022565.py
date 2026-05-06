def _net(self):
        """Record Network usage."""
        data = psutil.network_io_counters(pernic=True)
        if self.device not in data:
            raise CommandError("Unknown device: {0}".format(self.device))

        # Network bytes sent
        value = data[self.device].bytes_sent
        metric("net-{0}-sent".format(self.device), value, category=self.category)
        gauge("net-{0}-sent".format(self.device), value)

        # Network bytes received
        value = data[self.device].bytes_recv
        metric("net-{0}-recv".format(self.device), value, category=self.category)