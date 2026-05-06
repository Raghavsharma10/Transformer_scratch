def get_xmlrpc_server(self):
        """
        Returns PyPI's XML-RPC server instance
        """
        check_proxy_setting()
        if os.environ.has_key('XMLRPC_DEBUG'):
            debug = 1
        else:
            debug = 0
        try:
            return xmlrpclib.Server(XML_RPC_SERVER, transport=ProxyTransport(), verbose=debug)
        except IOError:
            self.logger("ERROR: Can't connect to XML-RPC server: %s" \
                    % XML_RPC_SERVER)