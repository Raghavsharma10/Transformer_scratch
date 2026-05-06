def max_bit_rate(self):
        """
        Returns a tuple with the maximun upstream- and downstream-rate
        of the given connection. The rate is given in bits/sec.
        """
        status = self.fc.call_action('WANCommonInterfaceConfig',
                                     'GetCommonLinkProperties')
        downstream = status['NewLayer1DownstreamMaxBitRate']
        upstream = status['NewLayer1UpstreamMaxBitRate']
        return upstream, downstream