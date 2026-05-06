def tcpPorts(self, req, tag):
        """
        Create and return a L{PortScrollingFragment} for the L{TCPPort} items
        in site store.
        """
        f = PortScrollingFragment(
            self.store,
            TCPPort,
            (TCPPort.portNumber,
             TCPPort.interface,
             FactoryColumn(TCPPort.factory)))
        f.setFragmentParent(self)
        f.docFactory = webtheme.getLoader(f.fragmentName)
        return tag[f]