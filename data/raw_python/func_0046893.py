def _get_proxy_info(self, _=None):
        """Generate a ProxyInfo class from a connected SSH transport

        Args:
            _ (None): Ignored.  This is just here as the ProxyInfo spec requires it.


        Returns:
            SSHTunnelProxyInfo: A ProxyInfo with an active socket tunneled through SSH

        """
        # parse the fleet endpoint url, to establish a tunnel to that host
        (target_host, target_port, target_path) = self._endpoint_to_target(self._endpoint)

        # implement the proxy_info interface from httplib which requires
        # that we accept a scheme, and return a ProxyInfo object
        # we do :P
        # This is called once per request, so we keep this here
        # so that we can keep one ssh connection open, and allocate
        # new channels as needed per-request
        sock = None

        if target_path:
            sock = self._ssh_tunnel.forward_unix(path=target_path)
        else:
            sock = self._ssh_tunnel.forward_tcp(target_host, port=target_port)

        # Return a ProxyInfo class with this socket
        return SSHTunnelProxyInfo(sock=sock)