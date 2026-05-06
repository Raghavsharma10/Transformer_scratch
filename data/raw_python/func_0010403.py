def str_transmission_rate(self):
        """Returns a tuple of human readable transmission rates in bytes."""
        upstream, downstream = self.transmission_rate
        return (
            fritztools.format_num(upstream),
            fritztools.format_num(downstream)
        )