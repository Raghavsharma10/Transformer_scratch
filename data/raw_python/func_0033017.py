def indirect(self, interface):
        """
        Indirect the implementation of L{IWebViewer} to L{_AnonymousWebViewer}.
        """
        if interface == IWebViewer:
            return _AnonymousWebViewer(self.store)
        return super(AnonymousSite, self).indirect(interface)