def set_tunnels(self, tunnels):
        """
        set_tunnels(self, tunnels):
        """
        for tunnel_type, (tunnel, callback) in tunnels.iteritems():
            if tunnel is None:
                continue
            self.set_tunnel(tunnel_type, tunnel, callback)