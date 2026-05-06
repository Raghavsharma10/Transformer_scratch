def str_max_bit_rate(self):
        """
        Returns a human readable maximun upstream- and downstream-rate
        of the given connection. The rate is given in bits/sec.
        """
        upstream, downstream = self.max_bit_rate
        return (
            fritztools.format_rate(upstream, unit='bits'),
            fritztools.format_rate(downstream, unit ='bits')
        )