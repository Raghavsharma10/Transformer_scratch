def set_pause_param(self, autoneg, rx_pause, tx_pause):
        """
        Ethernet has flow control! The inter-frame pause can be adjusted, by
        auto-negotiation through an ethernet frame type with a simple two-field
        payload, and by setting it explicitly.

        http://en.wikipedia.org/wiki/Ethernet_flow_control
        """
        # create a struct ethtool_pauseparm
        # create a struct ifreq with its .ifr_data pointing at the above
        ecmd = array.array('B', struct.pack('IIII',
            ETHTOOL_SPAUSEPARAM, bool(autoneg), bool(rx_pause), bool(tx_pause)))
        buf_addr, _buf_len = ecmd.buffer_info()
        ifreq = struct.pack('16sP', self.name, buf_addr)
        fcntl.ioctl(sockfd, SIOCETHTOOL, ifreq)