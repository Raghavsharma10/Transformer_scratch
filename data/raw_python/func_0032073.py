def sslPorts(self, req, tag):
        """
        Create and return a L{PortScrollingFragment} for the L{SSLPort} items
        in the site store.
        """
        f = PortScrollingFragment(
            self.store,
            SSLPort,
            (SSLPort.portNumber,
             SSLPort.interface,
             CertificateColumn(SSLPort.certificatePath),
             FactoryColumn(SSLPort.factory)))
        f.setFragmentParent(self)
        f.docFactory = webtheme.getLoader(f.fragmentName)
        return tag[f]